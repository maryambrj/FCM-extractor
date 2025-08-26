import pandas as pd
import json
import numpy as np
import ast
import re
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from scipy.optimize import linear_sum_assignment
import os
import argparse
import sys
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.constants import EDGE_CONFIDENCE_THRESHOLD, USE_CONFIDENCE_FILTERING
from typing import Dict, List, Tuple, Optional


def build_cluster_texts(metadata_json):
    """
    Accepts dict-like metadata keyed by cluster ids (e.g., 'cluster_0').
    Returns:
      doc_texts: map usable under both keys: id -> text AND name -> text
      id_to_name: map id -> human-readable name
    """
    doc_texts = {}
    id_to_name = {}
    
    # If metadata is a dict of clusters
    if isinstance(metadata_json, dict) and "clusters" not in metadata_json:
        items = metadata_json.items()
        # print(f"Processing {len(items)} clusters as dict items")
    else:
        clusters_data = metadata_json.get("clusters", [])
        items = enumerate(clusters_data)
        # print(f"Processing {len(clusters_data)} clusters from list")

    for k, c in items:
        # Normalize to a common record
        cid   = str(c.get("id", k)).strip()
        cname = str(c.get("name", cid)).strip()
        concepts = c.get("concepts", []) or []
        summary  = c.get("summary", "")

        # Build a rich but short text (avoid banned fields like embeddings)
        # Keep top-N concepts for signal; tweak N if needed
        conc_text = ", ".join(concepts[:10])
        blob = " ".join([cname, summary, conc_text]).strip() or cname

        # Fill both keys so enrichment works regardless of whether FCM uses id or name
        doc_texts[cid] = blob
        doc_texts[cname] = blob
        id_to_name[cid] = cname

    # print(f"Built rich text for {len(doc_texts)//2} clusters (with both id and name keys)")
    return doc_texts, id_to_name


class ScoreCalculator:
    cache = {}
    def __init__(self, threshold, model_name, data, tp_scale=1, pp_scale=1.1, seed=42, doc_texts=None):  #pp 1
        # Set seeds for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        self.model_name = model_name
        self.data = data
        self.threshold = threshold
        self.tp_scale = tp_scale
        self.pp_scale = pp_scale
        self.doc_texts = doc_texts or {}  # cluster_id -> rich_text mapping
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding_side='left')
        self.model = AutoModel.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32  # <- force fp32
        )
        self.model.eval()              # <- disable dropout
        self.task_instruction = "Retrieve clusters that are semantically similar."
        
        try:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
            self.model = self.model.to(self.device)
        except RuntimeError as e:
            if "out of memory" in str(e) or "MPS backend out of memory" in str(e):
                print("GPU memory insufficient, falling back to CPU...")
                self.device = "cpu"
                self.model = self.model.to(self.device)
            else:
                raise e

    def mean_pool(self, last_hidden_states, attention_mask):
        mask = attention_mask.unsqueeze(-1).to(last_hidden_states.dtype)
        summed = (last_hidden_states * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    def _sanitize(self, x):
        """Sanitize input while preserving alignment - don't drop entries."""
        s = "" if x is None else str(x).strip()
        return s if s else "<EMPTY>"

    def embed_and_score(self, queries, documents, batch_size=4):
        """
        Compute similarity matrix between queries and documents.
        
        Returns:
            torch.Tensor: Similarity matrix of shape (len(queries), len(documents))
                         where result[i, j] is similarity between queries[i] and documents[j]
        """
        # Preserve alignment by sanitizing but not filtering out entries
        clean_queries = [self._sanitize(q) for q in queries]
        clean_documents = [self._sanitize(d) for d in documents]

        if not clean_queries or not clean_documents:
            return torch.zeros((len(queries), len(documents)))

        # Enrich *generated clusters* (queries) with metadata
        enriched_queries = [self.doc_texts.get(q, q) for q in clean_queries]
        


        text_prompt = lambda t: f"Instruct: {self.task_instruction}\nQuery: {t}"
        query_texts = [text_prompt(q) for q in enriched_queries]
        document_texts = clean_documents  # GT names stay plain

        query_embeddings = []
        for i in range(0, len(query_texts), batch_size):
            batch_texts = query_texts[i:i + batch_size]
            batch = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=4096,  
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**batch)
            batch_embeddings = self.mean_pool(outputs.last_hidden_state, batch['attention_mask'])
            query_embeddings.append(batch_embeddings)
            
            del batch, outputs, batch_embeddings
            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()

        document_embeddings = []
        for i in range(0, len(document_texts), batch_size):
            batch_texts = document_texts[i:i + batch_size]
            batch = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=4096, 
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**batch)
            batch_embeddings = self.mean_pool(outputs.last_hidden_state, batch['attention_mask'])
            document_embeddings.append(batch_embeddings)
            
            del batch, outputs, batch_embeddings
            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()

        if not query_embeddings or not document_embeddings:
            return torch.zeros((len(queries), len(documents)))
            
        query_embeddings = torch.cat(query_embeddings, dim=0)
        document_embeddings = torch.cat(document_embeddings, dim=0)
        
        # Robust normalization (no NaNs)
        query_embeddings = torch.nan_to_num(query_embeddings)
        document_embeddings = torch.nan_to_num(document_embeddings)

        # safe_l2_norm
        def safe_l2_norm(x):
            return x / torch.clamp(torch.norm(x, dim=1, keepdim=True), min=1e-6)

        query_embeddings = safe_l2_norm(query_embeddings)
        document_embeddings = safe_l2_norm(document_embeddings)

        similarity_matrix = torch.zeros(len(clean_queries), len(clean_documents), device=self.device)
        
        for i in range(0, len(clean_queries), batch_size):
            query_batch = query_embeddings[i:i + batch_size]
            for j in range(0, len(clean_documents), batch_size):
                doc_batch = document_embeddings[j:j + batch_size]
                batch_similarity = query_batch @ doc_batch.T
                similarity_matrix[i:i + batch_size, j:j + batch_size] = batch_similarity
                
                del batch_similarity
                if torch.cuda.is_available() and self.device == "cuda":
                    torch.cuda.empty_cache()

        return similarity_matrix

    def calculate_f1_score(self, tp, fp, fn, pp=None):
        if pp is None:
            denom = 2*tp + fp + fn
            return (2*tp)/denom if denom else 0.0
        denom = 2*tp + fp + fn + pp
        return (2*tp + pp)/denom if denom else 0.0
        
    def convert_matrix(self, df_matrix):
        values = df_matrix.values
        columns = df_matrix.columns
        index = df_matrix.index

        # Extract all non-zero entries (including negative values) as edges
        row_idx, col_idx = np.nonzero(values)

        sources_list = [index[r] for r in row_idx]   # rows → sources
        targets_list = [columns[c] for c in col_idx]  # cols → targets
        values_list = [float(values[r, c]) for r, c in zip(row_idx, col_idx)]

        return sources_list, targets_list, values_list

    def calculate_scores(self, gt_matrix, gen_matrix):
        self.gt_nodes_src, self.gt_nodes_tgt, self.gt_edge_dir = self.convert_matrix(gt_matrix)
        self.gen_nodes_src, self.gen_nodes_tgt, self.gen_edge_dir = self.convert_matrix(gen_matrix)
        
        # Create a more specific cache key that includes metadata state
        import hashlib, json
        cache_content = (
            tuple(self.gen_nodes_src), tuple(self.gt_nodes_src),
            tuple(self.gen_nodes_tgt), tuple(self.gt_nodes_tgt)
        )
        cache_hash = hashlib.md5(str(cache_content).encode()).hexdigest()[:8]
        docsig = hashlib.md5(json.dumps(self.doc_texts, sort_keys=True).encode()).hexdigest() if self.doc_texts else "nometa"
        cache_key = f"{self.data}_{self.model_name}_{cache_hash}_{docsig}"
        
        if cache_key not in type(self).cache:
            type(self).cache[cache_key] = {}
            
        if 'src' not in type(self).cache[cache_key]:
            # embed_and_score(queries, documents) returns (queries × documents) 
            # Call as (gen, gt) to get (gen × gt) similarity matrix
            src_scores = self.embed_and_score(self.gen_nodes_src, self.gt_nodes_src, getattr(self, 'batch_size', 2))
            type(self).cache[cache_key]['src'] = src_scores.detach().cpu().numpy()

        if 'tgt' not in type(self).cache[cache_key]:
            # embed_and_score(queries, documents) returns (queries × documents)
            # Call as (gen, gt) to get (gen × gt) similarity matrix
            tgt_scores = self.embed_and_score(self.gen_nodes_tgt, self.gt_nodes_tgt, getattr(self, 'batch_size', 2))
            type(self).cache[cache_key]['tgt'] = tgt_scores.detach().cpu().numpy()

        all_scores_src = np.array(type(self).cache[cache_key]['src'])
        all_scores_tgt = np.array(type(self).cache[cache_key]['tgt'])

        # Handle empty-edge cases explicitly
        if len(self.gen_edge_dir) == 0 or len(self.gt_edge_dir) == 0:
            self.TP = self.PP = 0
            self.FP = len(self.gen_edge_dir)
            self.FN = len(self.gt_edge_dir)
            
            print("TP, PP, FP, FN:", self.TP, self.PP, self.FP, self.FN)

            TP_scaled = self.TP * self.tp_scale
            PP_scaled = self.PP * self.pp_scale

            F1 = self.calculate_f1_score(TP_scaled, self.FP, self.FN, PP_scaled)
            
            model_score = pd.DataFrame({
                'Model': [self.model_name],
                'data': [self.data],
                'F1': [F1],
                'TP': [self.TP],
                'PP': [self.PP],
                'FP': [self.FP],
                'FN': [self.FN],
                'threshold': [self.threshold],
                'tp_scale': [self.tp_scale],
                'pp_scale': [self.pp_scale],
                'gt_nodes': [len(set(self.gt_nodes_src + self.gt_nodes_tgt))],
                'gt_edges': [len(self.gt_edge_dir)],
                'gen_nodes': [len(set(self.gen_nodes_src + self.gen_nodes_tgt))],
                'gen_edges': [len(self.gen_edge_dir)]
            })
            
            self.scores_df = model_score
            return model_score

        # Build masks and scores for bipartite matching
        mask = (all_scores_src >= self.threshold) & (all_scores_tgt >= self.threshold)
        combined = (all_scores_src + all_scores_tgt) / 2.0  # gen × gt

        # 1-to-1 assignment via Hungarian algorithm with TP tie-breaking bias
        LARGE = 1e6
        sign_mismatch = (np.sign(self.gen_edge_dir)[:, None] != np.sign(self.gt_edge_dir)[None, :])
        cost = np.where(mask, -combined + 1e-6*sign_mismatch, LARGE)  # maximize combined, bias toward TP
        row_ind, col_ind = linear_sum_assignment(cost)

        # Keep only valid pairs (passed thresholds)
        matched_pairs = [(int(i), int(j)) for i, j in zip(row_ind, col_ind) if mask[int(i), int(j)]]

        # Classify matches
        TP = 0
        PP = 0
        for i, j in matched_pairs:
            gen_sign = np.sign(self.gen_edge_dir[i])
            gt_sign  = np.sign(self.gt_edge_dir[j])
            if gen_sign == gt_sign:
                TP += 1
            else:
                PP += 1

        matched_gen = {i for i, _ in matched_pairs}
        matched_gt  = {j for _, j in matched_pairs}

        FP = len(self.gen_edge_dir) - len(matched_gen)
        FN = len(self.gt_edge_dir) - len(matched_gt)

        self.TP, self.PP, self.FP, self.FN = TP, PP, FP, FN
        
        # Sanity assertions to verify identity constraints
        assert self.TP + self.PP + self.FP == len(self.gen_edge_dir), \
            f"Gen edge identity violated: {self.TP + self.PP + self.FP} != {len(self.gen_edge_dir)}"
        assert self.TP + self.PP + self.FN == len(self.gt_edge_dir), \
            f"GT edge identity violated: {self.TP + self.PP + self.FN} != {len(self.gt_edge_dir)}"
        
        # Identity constraints are now guaranteed by construction:
        # TP + PP + FN = len(gt_edge_dir)
        # TP + PP + FP = len(gen_edge_dir)

        # print("Kept predicted edges after filtering:", len(self.gen_edge_dir))
        # print("GT edges:", len(self.gt_edge_dir))
        # print("Mean src/tgt sims above threshold:",
        #       (all_scores_src >= self.threshold).mean(),
        #       (all_scores_tgt >= self.threshold).mean())
        print("TP, PP, FP, FN:", self.TP, self.PP, self.FP, self.FN)

        TP_scaled = self.TP * self.tp_scale
        PP_scaled = self.PP * self.pp_scale

        F1 = self.calculate_f1_score(TP_scaled, self.FP, self.FN, PP_scaled)
        
        model_score = pd.DataFrame({
            'Model': [self.model_name],
            'data': [self.data],
            'F1': [F1],
            'TP': [self.TP],
            'PP': [self.PP],
            'FP': [self.FP],
            'FN': [self.FN],
            'threshold': [self.threshold],
            'tp_scale': [self.tp_scale],
            'pp_scale': [self.pp_scale],
            'gt_nodes': [len(set(self.gt_nodes_src + self.gt_nodes_tgt))],
            'gt_edges': [len(self.gt_edge_dir)],
            'gen_nodes': [len(set(self.gen_nodes_src + self.gen_nodes_tgt))],
            'gen_edges': [len(self.gen_edge_dir)]
        })
        
        self.scores_df = model_score

        return model_score


def load_fcm_data(gt_path, gen_path, metadata_path=None):
    gt_matrix = pd.read_csv(gt_path, index_col=0)
    
    # Convert empty strings to zeros in ground truth matrix
    gt_matrix = gt_matrix.replace('', 0)
    gt_matrix = gt_matrix.replace('""', 0)
    
    # Ensure all values are numeric
    gt_matrix = gt_matrix.apply(pd.to_numeric, errors='coerce').fillna(0)

    with open(gen_path, "r") as f:
        gen_json = json.load(f)

    doc_texts, id_to_name = {}, {}
    if metadata_path and os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            metadata_json = json.load(f)
        doc_texts, id_to_name = build_cluster_texts(metadata_json)
        print(f"Loaded metadata for {len(id_to_name)} clusters")

    gen_matrix = json_to_matrix(gen_json, id_to_name=id_to_name)
    return gt_matrix, gen_matrix, doc_texts


def json_to_matrix(json_data: Dict, id_to_name=None) -> pd.DataFrame:
    # Map node ids -> preferred node label
    # Prefer metadata name when available; otherwise keep whatever is in JSON
    def label_for(raw_id):
        raw_id = str(raw_id)
        if id_to_name and raw_id in id_to_name:
            return id_to_name[raw_id]
        return raw_id 

    edges = []
    for e in json_data["edges"]:
        if e.get("type") != "inter_cluster":
            continue
        conf = float(e.get("confidence", 0.0))
        if USE_CONFIDENCE_FILTERING and conf <= EDGE_CONFIDENCE_THRESHOLD:
            continue
        s = label_for(e["source"])
        t = label_for(e["target"])
        w = float(e["weight"])
        edges.append((s, t, w))

    # Build adjacency with correct orientation
    if not edges:
        nodes = set()
    else:
        nodes = set([s for s, _, _ in edges] + [t for _, t, _ in edges])
    nodes = sorted(nodes) or ["empty_graph"]
    mat = pd.DataFrame(0.0, index=nodes, columns=nodes)
    for s, t, w in edges:
        mat.loc[s, t] = w
    print(f"Created evaluation matrix with {len(edges)} edges")
    return mat


def main():
    parser = argparse.ArgumentParser(description='Score FCM against ground truth')
    parser.add_argument('--gt-path', required=True, help='Path to ground truth CSV')
    parser.add_argument('--gen-path', required=True, help='Path to generated FCM JSON')
    parser.add_argument('--threshold', type=float, default=0.6, help='Similarity threshold')
    parser.add_argument('--model-name', default='Qwen/Qwen3-Embedding-0.6B', help='Model name for scoring')
    parser.add_argument('--tp-scale', type=float, default=1.0, help='True positive scale')
    parser.add_argument('--pp-scale', type=float, default=1.1, help='Partial positive scale')
    parser.add_argument('--batch-size', type=int, default=2, help='Batch size for processing (lower = less memory)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--metadata-path', help='Path to cluster metadata JSON file (e.g., BD007_cluster_metadata.json)')
    
    args = parser.parse_args()
    
    data_name = os.path.splitext(os.path.basename(args.gt_path))[0]
    output_dir = os.path.dirname(args.gen_path)
    
    # Auto-construct metadata path if not provided
    metadata_path = args.metadata_path
    if not metadata_path:
        # Try to find metadata file based on dataset name
        potential_metadata_path = os.path.join(output_dir, f"{data_name}_cluster_metadata.json")
        print(f"Looking for metadata at: {potential_metadata_path}")
        if os.path.exists(potential_metadata_path):
            metadata_path = potential_metadata_path
            print(f"Auto-detected metadata file: {metadata_path}")
        else:
            print(f"No metadata file found at expected location: {potential_metadata_path}")
            metadata_path = None
    else:
        print(f"Using provided metadata path: {metadata_path}")
    
    final_metadata_path = metadata_path
    
    if not os.path.exists(args.gt_path):
        print(f"Error: Ground truth file {args.gt_path} not found!")
        return
    
    if not os.path.exists(args.gen_path):
        print(f"Error: Generated FCM file {args.gen_path} not found!")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    gt_matrix, gen_matrix, doc_texts = load_fcm_data(args.gt_path, args.gen_path, 
                                                    final_metadata_path)
    
    print(f"Ground truth matrix shape: {gt_matrix.shape}")
    print(f"Generated matrix shape: {gen_matrix.shape}")

    gen_matrix_output_path = os.path.join(output_dir, f"{data_name}_generated_matrix.csv")
    gen_matrix.to_csv(gen_matrix_output_path)
    print(f"Generated FCM matrix saved to: {gen_matrix_output_path}")
    
    scorer = ScoreCalculator(
        threshold=args.threshold,
        model_name=args.model_name,
        data=data_name,
        tp_scale=args.tp_scale,
        pp_scale=args.pp_scale,
        seed=args.seed,
        doc_texts=doc_texts
    )
    
    scorer.batch_size = args.batch_size
    
    print("Calculating F1 score...")
    result = scorer.calculate_scores(gt_matrix, gen_matrix)
    
    print("\n" + "="*50)
    print("FCM SCORING RESULTS")
    print("="*50)
    print(f"Model: {args.model_name}")
    print(f"Dataset: {data_name}")
    print(f"Threshold: {args.threshold}")
    print(f"TP Scale: {args.tp_scale}")
    print(f"PP Scale: {args.pp_scale}")
    print(f"F1 Score: {result['F1'].iloc[0]:.4f}")
    print(f"True Positives: {scorer.TP}")
    print(f"Partial Positives: {scorer.PP}")
    print(f"False Positives: {scorer.FP}")
    print(f"False Negatives: {scorer.FN}")
    print(f"Ground Truth Nodes: {result['gt_nodes'].iloc[0]}")
    print(f"Ground Truth Edges: {result['gt_edges'].iloc[0]}")
    print(f"Generated Nodes: {result['gen_nodes'].iloc[0]}")
    print(f"Generated Edges: {result['gen_edges'].iloc[0]}")
    print("="*50)
    
    output_file = os.path.join(output_dir, f"{data_name}_scoring_results.csv")
    result.to_csv(output_file, index=False)
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
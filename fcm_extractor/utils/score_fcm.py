import pandas as pd
import json
import numpy as np
import ast
import re
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import os
import argparse
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.constants import EVALUATION_INCLUDE_INTRA_CLUSTER_EDGES, EVALUATION_INCLUDE_INTRA_CLUSTER_NODES
from typing import Dict, List, Tuple


class ScoreCalculator:
    cache = {}
    def __init__(self, threshold, model_name, data, tp_scale=1, pp_scale=1.1):  #pp 1
        self.model_name = model_name
        self.data = data
        self.threshold = threshold
        self.tp_scale = tp_scale
        self.pp_scale = pp_scale
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding_side='left')
        self.model = AutoModel.from_pretrained(self.model_name)
        self.task_instruction = "Retrieve concepts that are semantically similar."
        
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

    def last_token_pool(self, last_hidden_states, attention_mask):
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[
                torch.arange(batch_size, device=last_hidden_states.device),
                sequence_lengths
            ]

    def embed_and_score(self, queries, documents, batch_size=4):
        clean_queries = [str(q).strip() for q in queries if q is not None and str(q).strip()]
        clean_documents = [str(d).strip() for d in documents if d is not None and str(d).strip()]

        if not clean_queries or not clean_documents:
            return torch.zeros((len(queries), len(documents)))

        text_prompt = lambda t: f"Instruct: {self.task_instruction}\nQuery: {t}"
        query_texts = [text_prompt(q) for q in clean_queries]
        # document_texts = [text_prompt(d) for d in clean_documents]
        document_texts = clean_documents

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
            batch_embeddings = self.last_token_pool(outputs.last_hidden_state, batch['attention_mask'])
            query_embeddings.append(batch_embeddings)
            
            del batch, outputs, batch_embeddings
            if self.device != "cpu":
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
            batch_embeddings = self.last_token_pool(outputs.last_hidden_state, batch['attention_mask'])
            document_embeddings.append(batch_embeddings)
            
            del batch, outputs, batch_embeddings
            if self.device != "cpu":
                torch.cuda.empty_cache()

        if not query_embeddings or not document_embeddings:
            return torch.zeros((len(queries), len(documents)))
            
        query_embeddings = torch.cat(query_embeddings, dim=0)
        document_embeddings = torch.cat(document_embeddings, dim=0)
        
        query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
        document_embeddings = F.normalize(document_embeddings, p=2, dim=1)

        similarity_matrix = torch.zeros(len(clean_queries), len(clean_documents), device=self.device)
        
        for i in range(0, len(clean_queries), batch_size):
            query_batch = query_embeddings[i:i + batch_size]
            for j in range(0, len(clean_documents), batch_size):
                doc_batch = document_embeddings[j:j + batch_size]
                batch_similarity = query_batch @ doc_batch.T
                similarity_matrix[i:i + batch_size, j:j + batch_size] = batch_similarity
                
                del batch_similarity
                if self.device != "cpu":
                    torch.cuda.empty_cache()

        return similarity_matrix

    def calculate_f1_score(self, tp, fp, fn, pp=None):
        if pp is None:
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        else:
            if (2*tp + fp + fn + pp) == 0:
                return 0
            else:
                f1_score = (2*tp + pp) / (2*tp + fp + fn + pp)
        
        return f1_score
        
    def convert_matrix(self, df_matrix):
        values = df_matrix.values
        columns = df_matrix.columns
        index = df_matrix.index

        row_idx, col_idx = np.nonzero(values)

        sources_list = [columns[c] for c in col_idx]
        targets_list = [index[r] for r in row_idx]
        values_list = [int(values[r, c]) for r, c in zip(row_idx, col_idx)]

        return sources_list, targets_list, values_list

    def calculate_scores(self, gt_matrix, gen_matrix):
        self.gt_nodes_src, self.gt_nodes_tgt, self.gt_edge_dir = self.convert_matrix(gt_matrix)
        self.gen_nodes_src, self.gen_nodes_tgt, self.gen_edge_dir = self.convert_matrix(gen_matrix)
        
        if self.data not in self.cache:
            ScoreCalculator.cache[self.data] = {}

        if self.model_name not in self.cache[self.data]:
            ScoreCalculator.cache[self.data][self.model_name] = {}
            
        if 'src' not in self.cache[self.data][self.model_name]:
            try:
                ScoreCalculator.cache[self.data][self.model_name]['src'] = self.embed_and_score(self.gen_nodes_src, self.gt_nodes_src, getattr(self, 'batch_size', 2)).cpu().numpy()
            except:
                ScoreCalculator.cache[self.data][self.model_name]['src'] = self.embed_and_score(self.gen_nodes_src, self.gt_nodes_src, getattr(self, 'batch_size', 2))

        if 'tgt' not in self.cache[self.data][self.model_name]:
            try:
                ScoreCalculator.cache[self.data][self.model_name]['tgt'] = self.embed_and_score(self.gen_nodes_tgt, self.gt_nodes_tgt, getattr(self, 'batch_size', 2)).cpu().numpy()
            except:
                ScoreCalculator.cache[self.data][self.model_name]['tgt'] = self.embed_and_score(self.gen_nodes_tgt, self.gt_nodes_tgt, getattr(self, 'batch_size', 2))

        all_scores_src = np.array(ScoreCalculator.cache[self.data][self.model_name]['src'])
        all_scores_tgt = np.array(ScoreCalculator.cache[self.data][self.model_name]['tgt'])

        # Debug: Print shapes
        print(f"Debug - Generated edges: {len(self.gen_edge_dir)}")
        print(f"Debug - GT edges: {len(self.gt_edge_dir)}")
        print(f"Debug - all_scores_src shape: {all_scores_src.shape}")
        print(f"Debug - all_scores_tgt shape: {all_scores_tgt.shape}")

        # For each generated edge, find the best matching GT edge
        gen_has_tp = np.zeros(len(self.gen_edge_dir), dtype=bool)
        gen_has_pp = np.zeros(len(self.gen_edge_dir), dtype=bool)
        
        for i in range(len(self.gen_edge_dir)):
            if i >= all_scores_src.shape[0]:
                print(f"Warning: Skipping generated edge {i}, out of bounds for scores matrix")
                break
                
            best_match_score = -1
            best_match_j = -1
            best_is_tp = False
            
            # Find best matching GT edge for this generated edge
            for j in range(len(self.gt_edge_dir)):
                if j >= all_scores_src.shape[1]:
                    print(f"Warning: Skipping GT edge {j}, out of bounds for scores matrix")
                    break
                    
                src_score = all_scores_src[i, j]
                tgt_score = all_scores_tgt[i, j]
                
                # Both source and target must be above threshold
                if src_score >= self.threshold and tgt_score >= self.threshold:
                    # Combined similarity score
                    combined_score = (src_score + tgt_score) / 2
                    
                    if combined_score > best_match_score:
                        best_match_score = combined_score
                        best_match_j = j
                        # Check if directions (weights) match
                        best_is_tp = (self.gen_edge_dir[i] == self.gt_edge_dir[j])
            
            # Classify based on best match
            if best_match_j >= 0:  # Found a valid match
                if best_is_tp:
                    gen_has_tp[i] = True
                else:
                    gen_has_pp[i] = True

        self.TP = np.sum(gen_has_tp)
        self.PP = np.sum(gen_has_pp)
        self.FP = len(self.gen_edge_dir) - self.TP - self.PP

        # For FN calculation: check which GT edges were not matched by any generated edge
        gt_has_match = np.zeros(len(self.gt_edge_dir), dtype=bool)
        
        for j in range(len(self.gt_edge_dir)):
            if j >= all_scores_src.shape[1]:
                print(f"Warning: Skipping GT edge {j} in FN calculation, out of bounds")
                break
                
            best_match_score = -1
            
            # Find if any generated edge matches this GT edge well enough
            for i in range(len(self.gen_edge_dir)):
                if i >= all_scores_src.shape[0]:
                    break
                    
                src_score = all_scores_src[i, j]
                tgt_score = all_scores_tgt[i, j]
                
                if src_score >= self.threshold and tgt_score >= self.threshold:
                    combined_score = (src_score + tgt_score) / 2
                    if combined_score > best_match_score:
                        best_match_score = combined_score
            
            # If we found any match above threshold, this GT edge is covered
            if best_match_score >= self.threshold:
                gt_has_match[j] = True
        
        self.FN = len(self.gt_edge_dir) - np.sum(gt_has_match)

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


def load_fcm_data(gt_path: str, gen_path: str, include_intra_cluster_edges: bool = EVALUATION_INCLUDE_INTRA_CLUSTER_EDGES,
                  include_intra_cluster_nodes: bool = EVALUATION_INCLUDE_INTRA_CLUSTER_NODES) -> Tuple[pd.DataFrame, pd.DataFrame]:
    
    print(f"Loading ground truth from: {gt_path}")
    gt_matrix = pd.read_csv(gt_path, index_col=0)
    
    print(f"Loading generated FCM from: {gen_path}")
    with open(gen_path, 'r') as f:
        gen_json = json.load(f)
    
    gen_matrix = json_to_matrix(gen_json, include_intra_cluster_edges, include_intra_cluster_nodes)
    
    return gt_matrix, gen_matrix


def json_to_matrix(json_data: Dict, include_intra_cluster_edges: bool = EVALUATION_INCLUDE_INTRA_CLUSTER_EDGES, 
                   include_intra_cluster_nodes: bool = EVALUATION_INCLUDE_INTRA_CLUSTER_NODES) -> pd.DataFrame:
    node_concepts = {}
    for node in json_data['nodes']:
        node_id = node['id']
        concepts = node['concepts']
        
        if include_intra_cluster_nodes:
            if isinstance(concepts, list):
                for i, concept in enumerate(concepts):
                    node_concepts[f"{node_id}_{i}"] = concept.strip() if concept else f"node_{node_id}_{i}"
            elif isinstance(concepts, str):
                for i, concept in enumerate(concepts.split(',')):
                    node_concepts[f"{node_id}_{i}"] = concept.strip() if concept else f"node_{node_id}_{i}"
            else:
                node_concepts[node_id] = f"node_{node_id}"
        else:
            if isinstance(concepts, list):
                node_concepts[node_id] = concepts[0].strip() if concepts else f"node_{node_id}"
            elif isinstance(concepts, str):
                node_concepts[node_id] = concepts.split(',')[0].strip() if concepts else f"node_{node_id}"
            else:
                node_concepts[node_id] = f"node_{node_id}"
    
    edges = []
    for edge in json_data['edges']:
        edge_type = edge.get('type', 'inter_cluster')
        
        if edge_type == 'intra_cluster' and not include_intra_cluster_edges:
            continue
            
        source_id = edge['source']
        target_id = edge['target']
        weight = edge['weight']
        
        if include_intra_cluster_nodes and edge_type == 'intra_cluster':
            source_node = next((node for node in json_data['nodes'] if node['id'] == source_id), None)
            target_node = next((node for node in json_data['nodes'] if node['id'] == target_id), None)
            
            if source_node and target_node:
                source_concepts = source_node['concepts'] if isinstance(source_node['concepts'], list) else source_node['concepts'].split(',')
                target_concepts = target_node['concepts'] if isinstance(target_node['concepts'], list) else target_node['concepts'].split(',')
                
                for src_concept in source_concepts:
                    for tgt_concept in target_concepts:
                        if src_concept.strip() and tgt_concept.strip():
                            edges.append({
                                'source': src_concept.strip(),
                                'target': tgt_concept.strip(),
                                'weight': weight,
                                'type': edge_type
                            })
        else:
            source_name = node_concepts.get(source_id, f"node_{source_id}")
            target_name = node_concepts.get(target_id, f"node_{target_id}")
            
            edges.append({
                'source': source_name,
                'target': target_name,
                'weight': weight,
                'type': edge_type
            })
    
    if not edges:
        all_nodes = list(node_concepts.values())
        if not all_nodes:
            all_nodes = ['empty_graph']
        matrix = pd.DataFrame(0, index=all_nodes, columns=all_nodes)
    else:
        all_nodes = list(set([edge['source'] for edge in edges] + [edge['target'] for edge in edges]))
        matrix = pd.DataFrame(0, index=all_nodes, columns=all_nodes)
        
        for edge in edges:
            matrix.loc[edge['source'], edge['target']] = edge['weight']
    
    print(f"Created evaluation matrix with {len(edges)} edges ({len([e for e in edges if e.get('type') == 'inter_cluster'])} inter-cluster, {len([e for e in edges if e.get('type') == 'intra_cluster'])} intra-cluster)")
    
    return matrix


def main():
    parser = argparse.ArgumentParser(description='Score FCM against ground truth')
    parser.add_argument('--gt-path', required=True, help='Path to ground truth CSV')
    parser.add_argument('--gen-path', required=True, help='Path to generated FCM JSON')
    parser.add_argument('--threshold', type=float, default=0.6, help='Similarity threshold')
    parser.add_argument('--model-name', default='Qwen/Qwen3-Embedding-0.6B', help='Model name for scoring')
    parser.add_argument('--tp-scale', type=float, default=1.0, help='True positive scale')
    parser.add_argument('--pp-scale', type=float, default=1.1, help='Partial positive scale')
    parser.add_argument('--batch-size', type=int, default=2, help='Batch size for processing (lower = less memory)')
    parser.add_argument('--include-intra-cluster-edges', action='store_true', help='Include intra-cluster edges in evaluation')
    parser.add_argument('--include-intra-cluster-nodes', action='store_true', help='Include intra-cluster nodes in evaluation')
    
    args = parser.parse_args()
    
    data_name = os.path.splitext(os.path.basename(args.gt_path))[0]
    output_dir = os.path.dirname(args.gen_path)
    
    if not os.path.exists(args.gt_path):
        print(f"Error: Ground truth file {args.gt_path} not found!")
        return
    
    if not os.path.exists(args.gen_path):
        print(f"Error: Generated FCM file {args.gen_path} not found!")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    gt_matrix, gen_matrix = load_fcm_data(args.gt_path, args.gen_path, 
                                         args.include_intra_cluster_edges, 
                                         args.include_intra_cluster_nodes)
    
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
        pp_scale=args.pp_scale
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
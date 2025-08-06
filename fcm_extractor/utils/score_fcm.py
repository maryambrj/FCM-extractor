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
from pathlib import Path


def load_ground_truth_matrix(gt_path):
    """Load ground truth matrix from CSV file"""
    df = pd.read_csv(gt_path, index_col=0, encoding='utf-8-sig')  # Handle BOM
    return df


def load_generated_matrix_from_json(json_path):
    """Load generated FCM from JSON file and convert to adjacency matrix"""
    with open(json_path, 'r') as f:
        fcm_data = json.load(f)
    
    # Extract nodes
    nodes = [node['id'] for node in fcm_data['nodes']]
    
    # Create empty adjacency matrix
    matrix = pd.DataFrame(0, index=nodes, columns=nodes)
    
    # Fill in edges
    for edge in fcm_data['edges']:
        source = edge['source']
        target = edge['target']
        weight = edge.get('weight', 1.0)  # Default weight is 1.0
        
        if source in nodes and target in nodes:
            matrix.loc[target, source] = weight
    
    return matrix


def load_fcm_data(json_path):
    """Load FCM data from JSON file (alias for compatibility)"""
    return load_generated_matrix_from_json(json_path)


def json_to_matrix(json_path):
    """Convert JSON FCM to adjacency matrix (alias for compatibility)"""
    return load_generated_matrix_from_json(json_path)


def find_fcm_outputs(base_dir="fcm_outputs", pattern_prefix=""):
    """Find all FCM output directories and JSON files"""
    base_path = Path(base_dir)
    if not base_path.exists():
        # Try alternative paths
        alt_paths = ["../fcm_outputs", "../../fcm_outputs", "fcm_outputs_gpt-mini"]
        for alt_path in alt_paths:
            alt_base = Path(alt_path)
            if alt_base.exists():
                base_path = alt_base
                break
    
    fcm_files = []
    for subdir in base_path.iterdir():
        if subdir.is_dir() and (not pattern_prefix or subdir.name.startswith(pattern_prefix)):
            json_file = subdir / f"{subdir.name.split('-')[0]}_fcm.json"
            if json_file.exists():
                fcm_files.append(json_file)
    
    return fcm_files


def find_ground_truth_files(base_dir="ground_truth"):
    """Find all ground truth CSV files"""
    base_path = Path(base_dir)
    if not base_path.exists():
        # Try alternative paths
        alt_paths = ["../ground_truth", "../../ground_truth"]
        for alt_path in alt_paths:
            alt_base = Path(alt_path)
            if alt_base.exists():
                base_path = alt_base
                break
    
    return list(base_path.glob("*.csv"))



class ScoreCalculator:
    cache = {}
    def __init__(self, threshold, model_name, data, tp_scale=1, pp_scale=1, device=None):#,fp_scale=1,fn_scale=1):
        self.model_name = model_name
        self.data = data
        self.threshold = threshold
        self.tp_scale = tp_scale
        self.pp_scale = pp_scale
        # self.fp_scale=fp_scale
        # self.fn_scale=fn_scale
        
        # Check if we should force string-based similarity (for reliability)
        use_string_similarity = os.environ.get('FORCE_STRING_SIMILARITY', 'false').lower() == 'true'
        
        if use_string_similarity:
            print("Using string-based similarity (forced by environment variable)")
            self.model_loaded = False
            self.tokenizer = None
            self.model = None
        else:
            # Try to load embedding model
            try:
                print("Loading embedding model...")
                test_model_name = 'distilbert-base-uncased'
                
                # Try to load the model directly
                print(f"Loading tokenizer for {test_model_name}...")
                self.tokenizer = AutoTokenizer.from_pretrained(test_model_name)
                print(f"Loading model for {test_model_name}...")
                self.model = AutoModel.from_pretrained(test_model_name)
                self.model_loaded = True
                print("Embedding model loaded successfully")
                    
            except Exception as e:
                print(f"Cannot load embedding models reliably ({e}) - using string-based similarity")
                print(f"Error details: {type(e).__name__}: {str(e)}")
                self.model_loaded = False
                self.tokenizer = None
                self.model = None
        '''
        best params:
        QWEN 0.6B
            threshold: .6 
            tp_scale: 1.0
            pp_scale: 1.1
        QWEN 8B
            threshold: .5
            tp_scale: .4
            pp_scale: .1
        '''


        self.task_instruction = "Find concepts that represent variables capable of the same or similar quantitative or qualitative changes."
        
        # Only setup device if model is loaded
        if self.model_loaded:
            # Use specified device or auto-detect
            if device == "cpu":
                self.device = "cpu"
            else:
                self.device = "cpu"  # Default to CPU to avoid issues
                try:
                    if torch.cuda.is_available():
                        self.device = "cuda"
                    elif torch.backends.mps.is_available():
                        self.device = "mps"
                except Exception as e:
                    print(f"GPU detection failed, using CPU: {e}")
                
            self.model = self.model.to(self.device)
            print(f"Using device: {self.device}")
        else:
            self.device = "cpu"

    def _fallback_similarity(self, queries, documents):
        """Simple string-based similarity as fallback when embeddings fail"""
        import re
        from difflib import SequenceMatcher
        
        def normalize(s):
            s = str(s).lower().strip()
            s = re.sub(r'[^\w\s]', '', s)  # Remove punctuation
            s = ' '.join(s.split())  # Normalize whitespace
            return s
        
        def hybrid_similarity(a, b):
            norm_a = normalize(a)
            norm_b = normalize(b)
            
            # Exact match after normalization
            if norm_a == norm_b:
                return 1.0
            
            # One contains the other (high similarity)
            if norm_a in norm_b or norm_b in norm_a:
                return 0.9
            
            # Word overlap similarity
            words_a = set(norm_a.split())
            words_b = set(norm_b.split())
            
            if words_a and words_b:
                # Jaccard similarity for word overlap
                word_sim = len(words_a & words_b) / len(words_a | words_b)
                # Sequence similarity
                seq_sim = SequenceMatcher(None, norm_a, norm_b).ratio()
                return max(word_sim, seq_sim)
            
            # Sequence similarity only
            return SequenceMatcher(None, norm_a, norm_b).ratio()
        
        # Calculate similarity matrix
        similarity_matrix = np.zeros((len(documents), len(queries)))
        for i, doc in enumerate(documents):
            for j, query in enumerate(queries):
                similarity_matrix[i, j] = hybrid_similarity(doc, query)
        
        return torch.tensor(similarity_matrix, dtype=torch.float32)

    def mean_pooling(self, last_hidden_states, attention_mask):
        """Mean pooling - more stable than last token pooling"""
        token_embeddings = last_hidden_states
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def embed_and_score(self, queries, documents):
        clean_queries = [str(q).strip() for q in queries if q is not None and str(q).strip()]
        clean_documents = [str(d).strip() for d in documents if d is not None and str(d).strip()]

        if not clean_queries or not clean_documents:
            return torch.zeros((len(clean_documents), len(clean_queries)))
        
        # If no model loaded, use string-based similarity
        if not self.model_loaded:
            return self._fallback_similarity(clean_queries, clean_documents)

        try:
            # For the new model, don't use instruction formatting - just use the raw text
            # Combine for single batch embedding
            input_texts = clean_queries + clean_documents
            
            batch = self.tokenizer(
                input_texts,
                padding=True,
                truncation=True,
                max_length=512,  # Reduced from 8192 to avoid memory issues
                return_tensors="pt"
            ).to(self.device)

            # Forward pass
            with torch.no_grad():
                outputs = self.model(**batch)
                
            embeddings = self.mean_pooling(outputs.last_hidden_state, batch['attention_mask'])
            
            # Check for NaN or inf values
            if torch.isnan(embeddings).any() or torch.isinf(embeddings).any():
                print("Warning: NaN or Inf detected in embeddings")
                embeddings = torch.nan_to_num(embeddings, nan=0.0, posinf=1.0, neginf=-1.0)
                
            embeddings = F.normalize(embeddings, p=2, dim=1)
            
            # Check again after normalization
            if torch.isnan(embeddings).any() or torch.isinf(embeddings).any():
                print("Warning: NaN or Inf detected after normalization")
                embeddings = torch.nan_to_num(embeddings, nan=0.0, posinf=1.0, neginf=-1.0)

            # Split query and document embeddings
            query_embeddings = embeddings[:len(clean_queries)]
            document_embeddings = embeddings[len(clean_queries):]

            # Cosine similarity
            similarity = query_embeddings @ document_embeddings.T
            
            # Check similarity matrix
            if torch.isnan(similarity).any() or torch.isinf(similarity).any():
                print("Warning: NaN or Inf detected in similarity matrix")
                similarity = torch.nan_to_num(similarity, nan=0.0, posinf=1.0, neginf=-1.0)
            
            return similarity.T
            
        except Exception as e:
            print(f"Error in embed_and_score: {e}")
            # Return zero matrix as fallback
            return torch.zeros((len(clean_documents), len(clean_queries)))

        
    
    def calculate_f1_score(self, tp, fp, fn, pp=None):
        if pp is not None and pp > 0:
            if (2*tp+pp)==0:
                return  0
            elif(2*tp+fp+fn+pp)==0:
                return 0
            else:
                f1_score = (2*tp+pp)/(2*tp+fp+fn+pp)
                return f1_score
        else:
            if tp == 0:
                return 0
            elif (2*tp+fp+fn) == 0:
                return 0
            else:
                f1_score = (2*tp)/(2*tp+fp+fn)
                return f1_score

        
    def convert_matrix(self,df_matrix):
        values = df_matrix.values
        columns = df_matrix.columns
        index = df_matrix.index

        # Get row and column indices where value is non-zero
        row_idx, col_idx = np.nonzero(values)

        # Build lists using advanced indexing (fast and order-preserving)
        sources_list = [columns[c] for c in col_idx]
        targets_list = [index[r] for r in row_idx]
        values_list  = [int(values[r, c]) for r, c in zip(row_idx, col_idx)]

        return sources_list, targets_list, values_list
                
    def calculated_node_scores(self,gt_nodes,gen_nodes):
        # Initialize cache if needed
        if self.data not in self.cache:
            ScoreCalculator.cache[self.data] = {}
        if self.model_name not in self.cache[self.data]:
            ScoreCalculator.cache[self.data][self.model_name] = {}
            
        if 'node' not in self.cache[self.data][self.model_name]:
            try:
                ScoreCalculator.cache[self.data][self.model_name]['node'] = self.embed_and_score(gt_nodes, gen_nodes).cpu().numpy()
            except:
                ScoreCalculator.cache[self.data][self.model_name]['node'] = self.embed_and_score(gt_nodes, gen_nodes)
                
        all_scores_node = ScoreCalculator.cache[self.data][self.model_name]['node']
        binary_node_scores = all_scores_node >= self.threshold

        gen_has_tp = np.any(binary_node_scores, axis=1)
        tp_mask = np.any(binary_node_scores, axis=0)

        self.TP = np.sum(tp_mask)
        self.FP = np.sum(~gen_has_tp)
        self.FN = np.sum(~tp_mask)

        F1 = self.calculate_f1_score(self.TP, self.FP, self.FN)

        self.scores_df = pd.DataFrame(columns=['Model', 'data', 'F1'])
        model_score = pd.DataFrame({'Model': [self.model_name], 'data': [self.data], 'F1': [F1]})
        self.scores_df = pd.concat([self.scores_df, model_score])

        return model_score







    def calculate_scores(self,gt_matrix,gen_matrix):
                
        self.gt_nodes_src,self.gt_nodes_tgt, self.gt_edge_dir=self.convert_matrix(gt_matrix)
        self.gen_nodes_src,  self.gen_nodes_tgt, self.gen_edge_dir=self.convert_matrix(gen_matrix)
        
        # Calculate scores separately for source and target nodes
        # collect cached results if already computed
        if self.data  not in self.cache:
            ScoreCalculator.cache[self.data] = {}

        if self.model_name not in self.cache[self.data]:
            ScoreCalculator.cache[self.data][self.model_name] = {}
        if 'src' not in self.cache[self.data][self.model_name]:
            try:
                ScoreCalculator.cache[self.data][self.model_name]['src'] = self.embed_and_score(self.gt_nodes_src, self.gen_nodes_src).cpu().numpy()
            except:
                ScoreCalculator.cache[self.data][self.model_name]['src'] = self.embed_and_score(self.gt_nodes_src, self.gen_nodes_src)

        if 'tgt' not in self.cache[self.data][self.model_name]:
            try:
                ScoreCalculator.cache[self.data][self.model_name]['tgt'] = self.embed_and_score(self.gt_nodes_tgt, self.gen_nodes_tgt).cpu().numpy()
            except:
                ScoreCalculator.cache[self.data][self.model_name]['tgt'] = self.embed_and_score(self.gt_nodes_tgt, self.gen_nodes_tgt)


        all_scores_src = np.array(ScoreCalculator.cache[self.data][self.model_name]['src'])
        all_scores_tgt = np.array(ScoreCalculator.cache[self.data][self.model_name]['tgt'])

        all_scores_dir = np.zeros((len(self.gen_edge_dir), len(self.gt_edge_dir)))
        for i in range(len(self.gen_edge_dir)):
            for j in range(len(self.gt_edge_dir)):
                if self.gen_edge_dir[i]==self.gt_edge_dir[j]:
                    all_scores_dir[i][j]=True
                else:
                    all_scores_dir[i][j]=False

        all_scores_dir=all_scores_dir.astype(bool)

        binary_matrix_src = all_scores_src >= self.threshold
        binary_matrix_tgt = all_scores_tgt >= self.threshold

        common_mask = binary_matrix_src & binary_matrix_tgt

        matching_edges = common_mask & all_scores_dir
        pp_edges = common_mask & ~all_scores_dir

        gen_has_tp = np.any(matching_edges, axis=1)
        gen_has_pp = np.any(pp_edges, axis=1) & ~gen_has_tp

        # Apply per-gen filtering to prevent overlap
        matching_edges = matching_edges & gen_has_tp[:, None]
        pp_edges = pp_edges & gen_has_pp[:, None]

        tp_mask = np.any(matching_edges, axis=0)
        pp_mask = np.any(pp_edges, axis=0) & ~tp_mask

        self.TP = np.sum(tp_mask)
        self.PP = np.sum(pp_mask)
        self.FP = np.sum(~(gen_has_tp | gen_has_pp))
        self.FN = np.sum(~(tp_mask | pp_mask))

        TP=self.TP*self.tp_scale
        PP=self.PP*self.pp_scale
        
        F1 = self.calculate_f1_score(TP, self.FP, self.FN, PP)
        self.scores_df = pd.DataFrame(columns=['Model', 'data', 'F1'])
        model_score=pd.DataFrame({'Model': [self.model_name], 'data': [self.data], 'F1': [F1]})
        self.scores_df = pd.concat([self.scores_df,model_score])

        return model_score
    


def evaluate_single_case(gt_file, gen_file, threshold=0.6, tp_scale=1.0, pp_scale=1.1, model_name="embedding_model", debug=False, device=None):
    """
    Evaluate a single FCM generation case
    
    Args:
        gt_file (str): Path to ground truth CSV file
        gen_file (str): Path to generated FCM JSON file  
        threshold (float): Similarity threshold
        tp_scale (float): True positive scaling factor
        pp_scale (float): Partial positive scaling factor
        model_name (str): Name identifier for the model
        debug (bool): Print debug information
    
    Returns:
        pd.DataFrame: Score results
    """
    # Load matrices
    gt_matrix = load_ground_truth_matrix(gt_file)
    gen_matrix = load_generated_matrix_from_json(gen_file)
    
    if debug:
        print(f"\nDEBUG INFO:")
        print(f"Ground truth matrix shape: {gt_matrix.shape}")
        print(f"Generated matrix shape: {gen_matrix.shape}")
        print(f"Ground truth nodes (first 10): {list(gt_matrix.columns[:10])}")
        print(f"Generated nodes (first 10): {list(gen_matrix.columns[:10])}")
        
        # Count non-zero edges
        gt_edges = (gt_matrix != 0).sum().sum()
        gen_edges = (gen_matrix != 0).sum().sum()
        print(f"Ground truth edges: {gt_edges}")
        print(f"Generated edges: {gen_edges}")
        
        # Check for potential node matches by looking at some examples
        print(f"\nSample node name comparison:")
        gt_sample = list(gt_matrix.columns[:5])
        gen_sample = list(gen_matrix.columns[:5])
        for gt_node in gt_sample:
            print(f"GT: '{gt_node}'")
        print("vs")
        for gen_node in gen_sample:
            print(f"Gen: '{gen_node}'")
    
    # Extract data name from file
    data_name = Path(gt_file).stem
    
    # Calculate scores
    scorer = ScoreCalculator(threshold, model_name, data_name, tp_scale, pp_scale, device=device)
    results = scorer.calculate_scores(gt_matrix, gen_matrix)
    
    if debug:
        print(f"TP: {scorer.TP}, FP: {scorer.FP}, FN: {scorer.FN}")
        if hasattr(scorer, 'PP'):
            print(f"PP: {scorer.PP}")
        
        # Show top similarity scores to understand what's being matched
        print(f"\nTop 10 node similarity scores (threshold: {threshold}):")
        try:
            # Get the source similarity scores
            src_scores = scorer.cache[data_name][model_name]['src']
            src_max_scores = np.max(src_scores, axis=1)
            src_indices = np.argsort(src_max_scores)[::-1][:10]
            
            print("Source node similarities:")
            for i, idx in enumerate(src_indices):
                if idx < len(scorer.gt_nodes_src) and idx < len(src_max_scores):
                    gt_node = scorer.gt_nodes_src[idx] if idx < len(scorer.gt_nodes_src) else "N/A"
                    score = src_max_scores[idx]
                    print(f"  {i+1}. GT: '{gt_node}' -> Max similarity: {score:.3f}")
        except Exception as e:
            print(f"Could not show similarity scores: {e}")
    
    return results


def evaluate_all_cases(gt_dir="ground_truth", gen_dir="fcm_outputs", 
                      threshold=0.6, tp_scale=1.0, pp_scale=1.1,
                      model_name="embedding_model"):
    """
    Evaluate all matching cases between ground truth and generated FCMs
    
    Args:
        gt_dir (str): Directory containing ground truth CSV files
        gen_dir (str): Directory containing generated FCM JSON files
        threshold (float): Similarity threshold
        tp_scale (float): True positive scaling factor  
        pp_scale (float): Partial positive scaling factor
        model_name (str): Name identifier for the model
        
    Returns:
        pd.DataFrame: Combined score results for all cases
    """
    all_results = []
    
    gt_files = find_ground_truth_files(gt_dir)
    gen_files = find_fcm_outputs(gen_dir)
    
    # Create mapping from case ID to files
    gt_map = {Path(f).stem: f for f in gt_files}
    gen_map = {Path(f).stem.split('_')[0]: f for f in gen_files}
    
    # Find matching cases
    matching_cases = set(gt_map.keys()) & set(gen_map.keys())
    
    print(f"Found {len(matching_cases)} matching cases: {sorted(matching_cases)}")
    
    for case_id in sorted(matching_cases):
        try:
            gt_file = gt_map[case_id]
            gen_file = gen_map[case_id]
            
            print(f"Evaluating {case_id}...")
            result = evaluate_single_case(gt_file, gen_file, threshold, tp_scale, pp_scale, 
                                        f"{model_name}_{case_id}")
            all_results.append(result)
            
        except Exception as e:
            print(f"Error evaluating {case_id}: {e}")
            continue
    
    if all_results:
        combined_results = pd.concat(all_results, ignore_index=True)
        return combined_results
    else:
        return pd.DataFrame(columns=['Model', 'data', 'F1'])


def main():
    """
    Command-line interface for FCM scoring system
    """
    parser = argparse.ArgumentParser(description='FCM Scoring System')
    parser.add_argument('--gt-path', type=str, help='Path to ground truth CSV file')
    parser.add_argument('gen_path', type=str, nargs='?', help='Path to generated FCM JSON file')
    parser.add_argument('--gen-path', type=str, dest='gen_path_flag', help='Path to generated FCM JSON file (alternative to positional arg)')
    parser.add_argument('--gt-dir', type=str, default='ground_truth', 
                       help='Directory containing ground truth CSV files')
    parser.add_argument('--gen-dir', type=str, default='fcm_outputs',
                       help='Directory containing generated FCM JSON files')
    parser.add_argument('--threshold', type=float, default=0.6,
                       help='Similarity threshold (default: 0.6)')
    parser.add_argument('--tp-scale', type=float, default=1.0,
                       help='True positive scaling factor (default: 1.0)')
    parser.add_argument('--pp-scale', type=float, default=1.1,
                       help='Partial positive scaling factor (default: 1.1)')
    parser.add_argument('--model-name', type=str, default='embedding_model',
                       help='Model name identifier (default: embedding_model)')
    parser.add_argument('--batch-size', type=int, help='Batch size (ignored, for compatibility)')
    parser.add_argument('--debug', action='store_true', help='Print debug information')
    parser.add_argument('--cpu-only', action='store_true', help='Force CPU usage instead of GPU')
    
    args = parser.parse_args()
    
    print("FCM Scoring System")
    print("==================")
    
    # Determine gen_path from either positional argument or flag
    gen_path = args.gen_path or args.gen_path_flag
    
    if args.gt_path and gen_path:
        # Single case evaluation
        print(f"\nEvaluating single case:")
        print(f"Ground truth: {args.gt_path}")
        print(f"Generated FCM: {gen_path}")
        
        try:
            device_to_use = "cpu" if args.cpu_only else None
            result = evaluate_single_case(
                args.gt_path,
                gen_path,
                threshold=args.threshold,
                tp_scale=args.tp_scale,
                pp_scale=args.pp_scale,
                model_name=args.model_name,
                debug=args.debug,
                device=device_to_use
            )
            print("\nResults:")
            print(result)
            print(f"\nF1 Score: {result['F1'].iloc[0]:.4f}")
            
        except Exception as e:
            print(f"Error: {e}")
            return 1
            
    else:
        # Batch evaluation
        print(f"\nBatch evaluation:")
        print(f"Ground truth directory: {args.gt_dir}")
        print(f"Generated FCM directory: {args.gen_dir}")
        
        try:
            results = evaluate_all_cases(
                gt_dir=args.gt_dir,
                gen_dir=args.gen_dir,
                threshold=args.threshold,
                tp_scale=args.tp_scale,
                pp_scale=args.pp_scale,
                model_name=args.model_name
            )
            
            if not results.empty:
                print(f"\nResults summary:")
                print(results)
                print(f"\nMean F1 Score: {results['F1'].mean():.4f}")
                print(f"Std F1 Score: {results['F1'].std():.4f}")
            else:
                print("No results found. Check your file paths.")
                
        except Exception as e:
            print(f"Error: {e}")
            return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
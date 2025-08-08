#!/usr/bin/env python3
"""
Simple string-based scorer that doesn't use embeddings
"""

import sys
sys.path.append('fcm_extractor')

from fcm_extractor.utils.score_fcm import load_ground_truth_matrix, load_generated_matrix_from_json
import pandas as pd
import numpy as np
from difflib import SequenceMatcher
import re

def string_similarity(a, b):
    """Calculate similarity between two strings"""
    # Normalize strings
    def normalize(s):
        s = str(s).lower().strip()
        s = re.sub(r'[^\w\s]', '', s)  # Remove punctuation
        s = ' '.join(s.split())  # Normalize whitespace
        return s
    
    norm_a = normalize(a)
    norm_b = normalize(b)
    
    # Try different similarity measures
    
    # 1. Exact match after normalization
    if norm_a == norm_b:
        return 1.0
    
    # 2. One contains the other
    if norm_a in norm_b or norm_b in norm_a:
        return 0.8
    
    # 3. Sequence similarity
    seq_sim = SequenceMatcher(None, norm_a, norm_b).ratio()
    
    # 4. Word overlap
    words_a = set(norm_a.split())
    words_b = set(norm_b.split())
    if words_a and words_b:
        word_overlap = len(words_a & words_b) / len(words_a | words_b)
        # Take max of sequence and word similarity
        return max(seq_sim, word_overlap)
    
    return seq_sim

def simple_score_fcm(gt_file, gen_file, threshold=0.6):
    """Score FCM using simple string similarity"""
    
    # Load matrices
    gt_matrix = load_ground_truth_matrix(gt_file)
    gen_matrix = load_generated_matrix_from_json(gen_file)
    
    print(f"GT matrix: {gt_matrix.shape}, Gen matrix: {gen_matrix.shape}")
    
    # Convert to edge lists
    def convert_matrix(df_matrix):
        values = df_matrix.values
        columns = df_matrix.columns
        index = df_matrix.index
        row_idx, col_idx = np.nonzero(values)
        sources = [columns[c] for c in col_idx]
        targets = [index[r] for r in row_idx]
        edge_values = [values[r, c] for r, c in zip(row_idx, col_idx)]
        return sources, targets, edge_values
    
    gt_src, gt_tgt, gt_vals = convert_matrix(gt_matrix)
    gen_src, gen_tgt, gen_vals = convert_matrix(gen_matrix)
    
    print(f"GT edges: {len(gt_src)}, Gen edges: {len(gen_src)}")
    
    # Build similarity matrices for source and target nodes
    print("Computing similarities...")
    
    # Create similarity matrix for sources
    src_similarities = np.zeros((len(gen_src), len(gt_src)))
    for i, gen_node in enumerate(gen_src):
        for j, gt_node in enumerate(gt_src):
            src_similarities[i, j] = string_similarity(gen_node, gt_node)
    
    # Create similarity matrix for targets
    tgt_similarities = np.zeros((len(gen_tgt), len(gt_tgt)))
    for i, gen_node in enumerate(gen_tgt):
        for j, gt_node in enumerate(gt_tgt):
            tgt_similarities[i, j] = string_similarity(gen_node, gt_node)
    
    # Direction similarity (sign matching)
    dir_similarities = np.zeros((len(gen_vals), len(gt_vals)))
    for i, gen_val in enumerate(gen_vals):
        for j, gt_val in enumerate(gt_vals):
            if np.sign(gen_val) == np.sign(gt_val):
                dir_similarities[i, j] = 1.0
            else:
                dir_similarities[i, j] = 0.0
    
    # Apply thresholds
    src_binary = src_similarities >= threshold
    tgt_binary = tgt_similarities >= threshold
    
    # Find matching edges (both source and target above threshold)
    matching_edges = src_binary & tgt_binary & dir_similarities.astype(bool)
    partial_edges = (src_binary & tgt_binary) & ~dir_similarities.astype(bool)
    
    # Calculate metrics
    gen_has_tp = np.any(matching_edges, axis=1)
    gen_has_pp = np.any(partial_edges, axis=1) & ~gen_has_tp
    
    tp_mask = np.any(matching_edges, axis=0)
    pp_mask = np.any(partial_edges, axis=0) & ~tp_mask
    
    TP = np.sum(tp_mask)
    PP = np.sum(pp_mask)
    FP = np.sum(~(gen_has_tp | gen_has_pp))
    FN = np.sum(~(tp_mask | pp_mask))
    
    print(f"TP: {TP}, PP: {PP}, FP: {FP}, FN: {FN}")
    
    # Calculate F1 score
    if TP + PP == 0:
        f1 = 0.0
    elif 2*TP + FP + FN + PP == 0:
        f1 = 0.0
    else:
        f1 = (2*TP + PP) / (2*TP + FP + FN + PP)
    
    print(f"F1 Score: {f1:.4f}")
    
    # Show some examples of matches
    print("\nTop matching edges:")
    max_scores = np.max(matching_edges.astype(float), axis=1)
    top_indices = np.argsort(max_scores)[::-1][:5]
    
    for idx in top_indices:
        if max_scores[idx] > 0:
            best_match = np.argmax(matching_edges[idx])
            src_sim = src_similarities[idx, best_match]
            tgt_sim = tgt_similarities[idx, best_match]
            print(f"  Gen: {gen_src[idx]} -> {gen_tgt[idx]} ({gen_vals[idx]})")
            print(f"  GT:  {gt_src[best_match]} -> {gt_tgt[best_match]} ({gt_vals[best_match]})")
            print(f"  Src sim: {src_sim:.3f}, Tgt sim: {tgt_sim:.3f}")
            print()
    
    return f1

if __name__ == "__main__":
    f1_score = simple_score_fcm(
        "ground_truth/BD007.csv", 
        "fcm_outputs/BD007-3/BD007_fcm.json",
        threshold=0.6
    )
    print(f"\nFinal F1 Score: {f1_score:.4f}")
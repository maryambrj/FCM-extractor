import numpy as np
import random
import re
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.constants import (
    EDGE_INFERENCE_MODEL, EDGE_INFERENCE_TEMPERATURE, 
    DEFAULT_INTER_CLUSTER_EDGE_PROMPT, EDGE_CONFIDENCE_THRESHOLD,
    ACO_MAX_ITERATIONS, ACO_SAMPLES_PER_ITERATION, ACO_EVAPORATION_RATE,
    ACO_INITIAL_PHEROMONE, ACO_CONVERGENCE_THRESHOLD, ACO_GUARANTEE_COVERAGE,
    ENABLE_INTRA_CLUSTER_EDGES
)
from src.models.llm_client import llm_client


@dataclass
class ACOEdge:
    source: str
    target: str
    pheromone: float
    confidence_history: List[float]
    rationale_history: List[str]
    weight_history: List[int] = None
    last_sampled: int = 0
    
    def __post_init__(self):
        if self.weight_history is None:
            self.weight_history = []


class ACOEdgeInference:
    def __init__(self, 
                 initial_pheromone: float = ACO_INITIAL_PHEROMONE,
                 evaporation_rate: float = ACO_EVAPORATION_RATE,
                 max_iterations: int = ACO_MAX_ITERATIONS,
                 samples_per_iteration: int = ACO_SAMPLES_PER_ITERATION,
                 convergence_threshold: float = ACO_CONVERGENCE_THRESHOLD):
        self.initial_pheromone = initial_pheromone
        self.evaporation_rate = evaporation_rate
        self.max_iterations = max_iterations
        self.samples_per_iteration = samples_per_iteration
        self.convergence_threshold = convergence_threshold
        
        self.pheromone_matrix: Dict[Tuple[str, str], ACOEdge] = {}
        self.cluster_names: List[str] = []
        self.clusters: Dict[str, List[str]] = {}
        self.text: str = ""
        self.cluster_metadata_manager = None
        
    def initialize_pheromones(self, clusters: Dict[str, List[str]]):
        self.clusters = clusters
        self.cluster_names = list(clusters.keys())
        
        print(f"Initializing ACO with {len(self.cluster_names)} clusters")
        print(f"   Possible edges: {len(self.cluster_names) * (len(self.cluster_names) - 1)}")
        
        for source in self.cluster_names:
            for target in self.cluster_names:
                if source != target:
                    edge_key = (source, target)
                    self.pheromone_matrix[edge_key] = ACOEdge(
                        source=source,
                        target=target,
                        pheromone=self.initial_pheromone,
                        confidence_history=[],
                        rationale_history=[],
                        weight_history=[]
                    )
    
    def sample_edges(self, iteration: int) -> List[Tuple[str, str]]:
        total_pheromone = sum(edge.pheromone for edge in self.pheromone_matrix.values())
        
        if iteration == 0 and ACO_GUARANTEE_COVERAGE:
            print(f"   First iteration: guaranteeing coverage of all {len(self.pheromone_matrix)} edges")
            edge_keys = list(self.pheromone_matrix.keys())
            
            samples_needed = min(self.samples_per_iteration, len(edge_keys))
            sampled_edges = np.random.choice(len(edge_keys), size=samples_needed, replace=False)
            return [edge_keys[i] for i in sampled_edges]
        
        if total_pheromone == 0:
            edge_keys = list(self.pheromone_matrix.keys())
            samples_needed = min(self.samples_per_iteration, len(edge_keys))
            sampled_edges = np.random.choice(len(edge_keys), size=samples_needed, replace=False)
            return [edge_keys[i] for i in sampled_edges]
        
        edge_keys = list(self.pheromone_matrix.keys())
        pheromone_values = [self.pheromone_matrix[edge].pheromone for edge in edge_keys]
        
        probabilities = np.array(pheromone_values) / total_pheromone
        samples_needed = min(self.samples_per_iteration, len(edge_keys))
        
        sampled_indices = np.random.choice(len(edge_keys), size=samples_needed, replace=False, p=probabilities)
        return [edge_keys[i] for i in sampled_indices]
    
    def query_llm_oracle(self, edge_pairs: List[Tuple[str, str]]) -> List[Dict]:
        if not edge_pairs:
            return []
        
        print(f"   Querying LLM oracle for {len(edge_pairs)} edge pairs...")
        
        pairs_text = []
        for i, (source, target) in enumerate(edge_pairs):
            source_concepts = self.clusters[source]
            target_concepts = self.clusters[target]
            
            source_str = ', '.join(source_concepts)
            target_str = ', '.join(target_concepts)
            
            pairs_text.append(f"Pair {i+1}: '{source_str}' and '{target_str}'")
        
        pairs_str = '\n'.join(pairs_text)
        
        prompt = DEFAULT_INTER_CLUSTER_EDGE_PROMPT.format(
            text=self.text,
            pairs=pairs_str
        )
        
        messages = [
            {"role": "system", "content": "You are an expert analyst building a Fuzzy Cognitive Map."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            content, _ = llm_client.chat_completion(EDGE_INFERENCE_MODEL, messages, EDGE_INFERENCE_TEMPERATURE)
            
            results = []
            lines = content.strip().split('\n')
            
            for i, line in enumerate(lines):
                if i >= len(edge_pairs):
                    break
                
                source, target = edge_pairs[i]
                result = {
                    'source': source,
                    'target': target,
                    'relationship': 'no relationship',
                    'confidence': 0.0,
                    'weight': 0,
                    'rationale': line.strip()
                }
                
                if '->' in line and '(' in line and ')' in line:
                    try:
                        parts = line.split('->')
                        if len(parts) == 2:
                            target_part = parts[1].strip()
                            if '(' in target_part and ')' in target_part:
                                target_concept = target_part.split('(')[0].strip()
                                confidence_part = target_part.split('(')[1].split(')')[0]
                                
                                if 'positive' in confidence_part:
                                    result['relationship'] = 'positive'
                                    result['weight'] = 1
                                elif 'negative' in confidence_part:
                                    result['relationship'] = 'negative'
                                    result['weight'] = -1
                                
                                confidence_match = re.search(r'0\.\d+', confidence_part)
                                if confidence_match:
                                    result['confidence'] = float(confidence_match.group())
                    except Exception as e:
                        print(f"      ⚠️ Error parsing line {i+1}: {e}")
                
                results.append(result)
            
            return results
            
        except Exception as e:
            print(f"      Error in LLM oracle query: {e}")
            return []
    
    def update_pheromones(self, iteration: int, oracle_results: List[Dict]):
        for result in oracle_results:
            source = result['source']
            target = result['target']
            confidence = result['confidence']
            weight = result['weight']
            rationale = result['rationale']
            
            edge_key = (source, target)
            if edge_key in self.pheromone_matrix:
                edge = self.pheromone_matrix[edge_key]
                
                edge.confidence_history.append(confidence)
                edge.rationale_history.append(rationale)
                edge.weight_history.append(weight)
                edge.last_sampled = iteration
                
                if confidence >= EDGE_CONFIDENCE_THRESHOLD:
                    pheromone_deposit = confidence * abs(weight)
                    edge.pheromone += pheromone_deposit
                    print(f"      {source} -> {target}: +{pheromone_deposit:.3f} (conf: {confidence:.2f})")
                else:
                    print(f"      {source} -> {target}: below threshold (conf: {confidence:.2f})")
        
        for edge in self.pheromone_matrix.values():
            edge.pheromone *= (1 - self.evaporation_rate)
    
    def check_convergence(self, iteration: int) -> bool:
        if iteration < 2:
            return False
        
        recent_pheromones = []
        for edge in self.pheromone_matrix.values():
            if edge.last_sampled >= iteration - 1:
                recent_pheromones.append(edge.pheromone)
        
        if not recent_pheromones:
            return False
        
        pheromone_std = np.std(recent_pheromones)
        pheromone_mean = np.mean(recent_pheromones)
        
        if pheromone_mean > 0:
            coefficient_of_variation = pheromone_std / pheromone_mean
            converged = coefficient_of_variation < self.convergence_threshold
            
            if converged:
                print(f"   Convergence detected (CV: {coefficient_of_variation:.3f} < {self.convergence_threshold})")
            
            return converged
        
        return False
    
    def extract_final_edges(self) -> List[Dict]:
        final_edges = []
        
        for edge_key, edge in self.pheromone_matrix.items():
            if edge.confidence_history:
                avg_confidence = np.mean(edge.confidence_history)
                avg_weight = np.mean(edge.weight_history) if edge.weight_history else 0
                
                if avg_confidence >= EDGE_CONFIDENCE_THRESHOLD and abs(avg_weight) > 0:
                    final_edges.append({
                        'source': edge.source,
                        'target': edge.target,
                        'weight': avg_weight,
                        'confidence': avg_confidence,
                        'pheromone': edge.pheromone,
                        'samples': len(edge.confidence_history),
                        'rationale': edge.rationale_history[-1] if edge.rationale_history else ""
                    })
        
        final_edges.sort(key=lambda x: x['pheromone'], reverse=True)
        return final_edges
    
    def infer_edges(self, clusters: Dict[str, List[str]], text: str,
                    cluster_metadata_manager = None) -> Tuple[List[Dict], List[Dict]]:
        self.text = text
        self.cluster_metadata_manager = cluster_metadata_manager
        
        print(f"Starting ACO edge inference with {len(clusters)} clusters")
        print(f"   Parameters: {self.max_iterations} iterations, {self.samples_per_iteration} samples/iter")
        
        self.initialize_pheromones(clusters)
        
        for iteration in range(self.max_iterations):
            print(f"   Iteration {iteration + 1}/{self.max_iterations}")
            
            sampled_edges = self.sample_edges(iteration)
            oracle_results = self.query_llm_oracle(sampled_edges)
            self.update_pheromones(iteration, oracle_results)
            
            if self.check_convergence(iteration):
                print(f"   Early stopping at iteration {iteration + 1}")
                break
        
        inter_cluster_edges = self.extract_final_edges()
        print(f"   ACO completed: {len(inter_cluster_edges)} edges found")
        
        intra_cluster_edges = []
        if ENABLE_INTRA_CLUSTER_EDGES:
            print(f"   Processing intra-cluster edges...")
            for cluster_name, concepts in clusters.items():
                if len(concepts) > 1:
                    cluster_edges = self._infer_intra_cluster_edges(cluster_name, concepts)
                    intra_cluster_edges.extend(cluster_edges)
        
        return inter_cluster_edges, intra_cluster_edges
    
    def _infer_intra_cluster_edges(self, cluster_name: str, concepts: List[str]) -> List[Dict]:
        if len(concepts) < 2:
            return []
        
        edge_pairs = []
        for i, concept1 in enumerate(concepts):
            for j, concept2 in enumerate(concepts[i+1:], i+1):
                edge_pairs.append((concept1, concept2))
        
        if not edge_pairs:
            return []
        
        pairs_text = []
        for i, (concept1, concept2) in enumerate(edge_pairs):
            pairs_text.append(f"Pair {i+1}: '{concept1}' and '{concept2}'")
        
        pairs_str = '\n'.join(pairs_text)
        
        from config.constants import DEFAULT_INTRA_CLUSTER_EDGE_PROMPT
        prompt = DEFAULT_INTRA_CLUSTER_EDGE_PROMPT.format(
            text=self.text,
            pairs=pairs_str
        )
        
        messages = [
            {"role": "system", "content": "You are an expert analyst building a Fuzzy Cognitive Map."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            content, _ = llm_client.chat_completion(EDGE_INFERENCE_MODEL, messages, EDGE_INFERENCE_TEMPERATURE)
            
            intra_edges = []
            lines = content.strip().split('\n')
            
            for i, line in enumerate(lines):
                if i >= len(edge_pairs):
                    break
                
                concept1, concept2 = edge_pairs[i]
                result = {
                    'source': concept1,
                    'target': concept2,
                    'relationship': 'no relationship',
                    'confidence': 0.0,
                    'weight': 0,
                    'cluster': cluster_name
                }
                
                if '->' in line and '(' in line and ')' in line:
                    try:
                        parts = line.split('->')
                        if len(parts) == 2:
                            target_part = parts[1].strip()
                            if '(' in target_part and ')' in target_part:
                                target_concept = target_part.split('(')[0].strip()
                                confidence_part = target_part.split('(')[1].split(')')[0]
                                
                                if 'positive' in confidence_part:
                                    result['relationship'] = 'positive'
                                    result['weight'] = 1
                                elif 'negative' in confidence_part:
                                    result['relationship'] = 'negative'
                                    result['weight'] = -1
                                
                                confidence_match = re.search(r'0\.\d+', confidence_part)
                                if confidence_match:
                                    result['confidence'] = float(confidence_match.group())
                    except Exception as e:
                        print(f"      Error parsing intra-cluster line {i+1}: {e}")
                
                if result['confidence'] >= EDGE_CONFIDENCE_THRESHOLD and abs(result['weight']) > 0:
                    intra_edges.append(result)
            
            return intra_edges
            
        except Exception as e:
            print(f"      Error in intra-cluster edge inference: {e}")
            return [] 
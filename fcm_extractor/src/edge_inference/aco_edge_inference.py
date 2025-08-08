"""
Ant Colony Optimization (ACO) based edge inference for FCM extraction.
Uses iterative sampling and reinforcement to discover causal relationships.
"""

import numpy as np
import random
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass
import sys
import os

# Add parent directories to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.constants import (
    EDGE_INFERENCE_MODEL, EDGE_INFERENCE_TEMPERATURE, 
    DEFAULT_INTER_CLUSTER_EDGE_PROMPT, EDGE_CONFIDENCE_THRESHOLD,
    ACO_MAX_ITERATIONS, ACO_SAMPLES_PER_ITERATION, ACO_EVAPORATION_RATE,
    ACO_INITIAL_PHEROMONE, ACO_CONVERGENCE_THRESHOLD, ACO_GUARANTEE_COVERAGE,
    ENABLE_INTRA_CLUSTER_EDGES
)
from src.models.llm_client import llm_client

# Note: Will use standard edge inference for intra-cluster edges to avoid circular import


@dataclass
class ACOEdge:
    """Represents an edge with pheromone level and metadata."""
    source: str
    target: str
    pheromone: float
    confidence_history: List[float]
    rationale_history: List[str]
    weight_history: List[int] = None  # Track positive/negative/neutral over time
    last_sampled: int = 0
    
    def __post_init__(self):
        if self.weight_history is None:
            self.weight_history = []


class ACOEdgeInference:
    """Ant Colony Optimization for edge inference."""
    
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
        self.cluster_metadata_manager = None  # Will be set during infer_edges
        
    def initialize_pheromones(self, clusters: Dict[str, List[str]]):
        """Initialize pheromone matrix for all cluster pairs."""
        self.clusters = clusters
        self.cluster_names = list(clusters.keys())
        
        print(f"🐜 Initializing ACO with {len(self.cluster_names)} clusters")
        print(f"   📊 Possible edges: {len(self.cluster_names) * (len(self.cluster_names) - 1)}")
        
        # Initialize all possible directed edges
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
        """Sample edges proportionally based on pheromone levels with coverage guarantee."""
        # Calculate total pheromone for normalization
        total_pheromone = sum(edge.pheromone for edge in self.pheromone_matrix.values())
        
        # In first iteration, ensure every edge gets tested at least once (if enabled)
        if iteration == 0 and ACO_GUARANTEE_COVERAGE:
            print(f"   🎯 First iteration: guaranteeing coverage of all {len(self.pheromone_matrix)} edges")
            edge_keys = list(self.pheromone_matrix.keys())
            
            # Sample evenly to cover all edges in the first iteration(s)
            samples_needed = min(self.samples_per_iteration, len(edge_keys))
            sampled_edges = np.random.choice(len(edge_keys), size=samples_needed, replace=False)
            return [edge_keys[i] for i in sampled_edges]
        
        # For subsequent iterations, use pheromone-based sampling
        if total_pheromone == 0 or len(self.pheromone_matrix) == 0:
            # If no pheromones yet or no edges, return empty list
            if len(self.pheromone_matrix) == 0:
                return []
            # Use uniform distribution
            probabilities = [1.0 / len(self.pheromone_matrix)] * len(self.pheromone_matrix)
        else:
            # Proportional selection based on pheromone levels
            probabilities = [edge.pheromone / total_pheromone for edge in self.pheromone_matrix.values()]
        
        # Sample edges proportionally (with replacement for consensus building)
        edge_keys = list(self.pheromone_matrix.keys())
        sampled_indices = np.random.choice(
            len(edge_keys), 
            size=self.samples_per_iteration, 
            replace=True,  # Allow same edge to be tested multiple times
            p=probabilities
        )
        
        sampled_edges = [edge_keys[i] for i in sampled_indices]
        
        # Count how many times each edge was selected
        edge_counts = {}
        for edge in sampled_edges:
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
        
        print(f"   🎯 Iteration {iteration + 1}: Sampled {len(set(sampled_edges))} unique edges from {len(sampled_edges)} selections")
        print(f"   📊 Most selected: {sorted(edge_counts.items(), key=lambda x: x[1], reverse=True)[:3]}")
        
        return list(set(sampled_edges))  # Return unique edges for this iteration
    
    def query_llm_oracle(self, edge_pairs: List[Tuple[str, str]]) -> List[Dict]:
        """Query the LLM for edge relationships."""
        if not edge_pairs:
            return []
        
        # Convert edge pairs to cluster pairs data format
        cluster_pairs_data = []
        for source, target in edge_pairs:
            source_concepts = self.clusters[source]
            target_concepts = self.clusters[target]
            cluster_pairs_data.append((source, source_concepts, target, target_concepts))
        
        # Extract relevant text from metadata if available
        relevant_text = self.text  # Default to full text
        
        if self.cluster_metadata_manager is not None:
            # Collect all relevant contexts from cluster metadata
            all_contexts = []
            seen_contexts = set()  # To avoid duplicates
            
            for source, _, target, _ in cluster_pairs_data:
                # Get contexts for this cluster pair
                contexts = self.cluster_metadata_manager.get_cluster_contexts_for_edge_inference(
                    source, target
                )
                
                # Add unique contexts
                for context in contexts:
                    if context not in seen_contexts:
                        seen_contexts.add(context)
                        all_contexts.append(context)
            
            if all_contexts:
                # Join contexts with clear separation
                relevant_text = "\n\n".join(all_contexts)
                print(f"    🔍 Using {len(all_contexts)} relevant contexts ({len(relevant_text)} chars) vs full text ({len(self.text)} chars)")
        
        # Format the pairs for the prompt
        pairs_text = ""
        for i, (source, source_concepts, target, target_concepts) in enumerate(cluster_pairs_data):
            pairs_text += (
                f"\n--- Pair {i+1} ---\n"
                f"Cluster A: \"{source}\" (Concepts: {', '.join(source_concepts[:3])})\n"
                f"Cluster B: \"{target}\" (Concepts: {', '.join(target_concepts[:3])})\n"
            )
        
        # Create the full prompt with relevant text only
        prompt = DEFAULT_INTER_CLUSTER_EDGE_PROMPT.format(text=relevant_text, pairs=pairs_text)
        messages = [{"role": "user", "content": prompt}]
        
        try:
            content, _ = llm_client.chat_completion(
                EDGE_INFERENCE_MODEL, messages, EDGE_INFERENCE_TEMPERATURE, max_tokens=4096
            )
            
            if not content.strip():
                print("    ⚠️ Empty LLM response")
                print(f"    📋 Debug - Model: {EDGE_INFERENCE_MODEL}")
                print(f"    🌡️ Debug - Temperature: {EDGE_INFERENCE_TEMPERATURE}")
                print(f"    📝 Debug - Prompt length: {len(str(messages))} chars")
                print(f"    🔢 Debug - Edge pairs: {len(edge_pairs)}")
                return []
            
            # DEBUG: Print raw LLM response
            print(f"    🔍 Raw LLM Response (first 500 chars):")
            print(f"    {content[:500]}...")
            
            # Parse responses using the correct format
            results = []
            lines = content.strip().split('\n')
            
            print(f"    📝 Found {len(lines)} response lines")
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Try to parse: "Pair X: concept1 -> concept2 (positive/negative, confidence: 0.XX)"
                relationship_pattern = re.compile(
                    r"pair\s+(\d+):\s*(.+?)\s*->\s*(.+?)\s*\(\s*(positive|negative)\s*,\s*confidence:\s*([0-1]\.\d+)\s*\)\s*",
                    re.IGNORECASE
                )
                
                # Try to parse: "Pair X: no relationship"
                no_relationship_pattern = re.compile(
                    r"pair\s+(\d+):\s*no\s+relationship",
                    re.IGNORECASE
                )
                
                match = relationship_pattern.match(line)
                if match:
                    pair_num = int(match.group(1)) - 1  # Convert to 0-based index
                    concept1 = match.group(2).strip().strip("'\"")
                    concept2 = match.group(3).strip().strip("'\"")
                    relationship = match.group(4).lower()
                    confidence = float(match.group(5))
                    
                    # Validate pair index
                    if 0 <= pair_num < len(edge_pairs):
                        source, target = edge_pairs[pair_num]
                        
                        # Determine weight based on relationship
                        weight = 1.0 if relationship == "positive" else -1.0
                        
                        print(f"    Pair {pair_num+1}: {source} -> {target} ({relationship}, conf={confidence})")
                        
                        results.append({
                            'source': source,
                            'target': target,
                            'confidence': confidence,
                            'weight': weight,
                            'rationale': f"Direct causal relationship: {source} {relationship}ly affects {target}"
                        })
                        
                elif no_relationship_pattern.match(line):
                    # Skip "no relationship" responses
                    continue
                else:
                    # Try to parse other formats or log unparseable lines
                    print(f"    ⚠️ Could not parse line: {line[:100]}...")
                    continue
            
            print(f"    ✅ Successfully parsed {len(results)} edges")
            return results
            
        except Exception as e:
            print(f"    ⚠️ LLM oracle error: {e}")
            return []
    
    def update_pheromones(self, iteration: int, oracle_results: List[Dict]):
        """Update pheromones for ALL tested edges with proper reinforcement."""
        # First: Evaporation for all edges
        for edge in self.pheromone_matrix.values():
            edge.pheromone *= (1 - self.evaporation_rate)
        
        # Second: Reinforcement based on LLM feedback
        reinforcement_count = 0
        
        for result in oracle_results:
            edge_key = (result['source'], result['target'])
            if edge_key in self.pheromone_matrix:
                edge = self.pheromone_matrix[edge_key]
                
                # Calculate reinforcement based on confidence
                confidence = result['confidence']
                
                # Even weak relationships get some reinforcement
                if confidence > 0.1:  # Very low threshold
                    delta = confidence * 0.2  # Stronger reinforcement
                    edge.pheromone += delta
                    reinforcement_count += 1
                    
                    # Store the evidence
                    edge.confidence_history.append(confidence)
                    edge.rationale_history.append(result['rationale'])
                    edge.weight_history.append(result['weight'])  # Store actual weight from LLM
                    edge.last_sampled = iteration
                
                # Also reinforce the reverse direction slightly if bidirectional
                reverse_key = (result['target'], result['source'])
                if reverse_key in self.pheromone_matrix and confidence > 0.3:
                    reverse_edge = self.pheromone_matrix[reverse_key]
                    reverse_edge.pheromone += confidence * 0.05  # Weaker reverse reinforcement
        
        print(f"   📈 Reinforced {reinforcement_count} edges based on LLM feedback")
        
        # Third: Show top pheromone edges for monitoring
        sorted_edges = sorted(self.pheromone_matrix.values(), key=lambda e: e.pheromone, reverse=True)
        top_3 = sorted_edges[:3]
        print(f"   🔝 Top pheromone edges: {[(e.source, e.target, f'{e.pheromone:.4f}') for e in top_3]}")
    
    def check_convergence(self, iteration: int) -> bool:
        """Check if the causal graph has stabilized."""
        if iteration < 2:  # Need at least 2 iterations for 3-iteration max
            return False
            
        # Get edges above convergence threshold
        strong_edges = [e for e in self.pheromone_matrix.values() 
                       if e.pheromone > self.convergence_threshold]
        
        if len(strong_edges) == 0:
            return False
            
        # Check if top edges have been consistently sampled
        stability_count = 0
        for edge in strong_edges[:15]:  # Top 15 strong edges
            if len(edge.confidence_history) >= 2:  # Tested at least twice
                # Check if confidence is consistent
                recent_confidences = edge.confidence_history[-2:]
                if abs(recent_confidences[0] - recent_confidences[1]) < 0.3:  # More tolerant
                    stability_count += 1
        
        convergence_ratio = stability_count / min(len(strong_edges), 15)
        print(f"   📊 Convergence check: {stability_count}/{min(len(strong_edges), 15)} stable edges ({convergence_ratio:.2f})")
        
        return convergence_ratio > 0.6  # 60% of top edges are stable
    
    def extract_final_edges(self) -> List[Dict]:
        """Extract final edges using consensus from multiple tests."""
        final_edges = []
        
        # Sort by pheromone strength
        sorted_edges = sorted(self.pheromone_matrix.values(), 
                            key=lambda e: e.pheromone, reverse=True)
        
        for edge in sorted_edges:
            if not edge.confidence_history:
                continue
                
            # Require multiple confirmations for strong edges
            num_tests = len(edge.confidence_history)
            avg_confidence = sum(edge.confidence_history) / num_tests
            max_confidence = max(edge.confidence_history)
            
            # Consensus-based filtering
            consensus_threshold = 0.2 if num_tests >= 2 else 0.3  # Lowered thresholds
            
            if (edge.pheromone > self.initial_pheromone * 1.5 and  # Lowered pheromone requirement
                avg_confidence > consensus_threshold and          # Lowered confidence threshold
                max_confidence > 0.3):                          # Lowered max confidence requirement
                
                # Get the best rationale and determine weight from history
                best_idx = edge.confidence_history.index(max_confidence)
                best_rationale = edge.rationale_history[best_idx] if edge.rationale_history else ""
                
                # Calculate weight based on weight history (more accurate)
                if edge.weight_history:
                    # Use the weight from the highest confidence response
                    weight = edge.weight_history[best_idx] if best_idx < len(edge.weight_history) else 1
                    
                    # Alternative: Could use majority vote from weight history
                    # positive_count = sum(1 for w in edge.weight_history if w > 0)
                    # negative_count = sum(1 for w in edge.weight_history if w < 0)
                    # weight = 1 if positive_count >= negative_count else -1
                else:
                    weight = 1  # Default positive if no history
                
                final_edges.append({
                    'source': edge.source,
                    'target': edge.target,
                    'weight': weight,  # Now uses calculated weight
                    'confidence': avg_confidence,
                    'max_confidence': max_confidence,
                    'pheromone': edge.pheromone,
                    'num_tests': num_tests,
                    'type': 'inter_cluster',
                    'rationale': best_rationale
                })
        
        print(f"   🎯 Consensus filtering: {len(final_edges)} edges from {sum(1 for e in self.pheromone_matrix.values() if e.confidence_history)} tested edges")
        
        return final_edges
    
    def infer_edges(self, clusters: Dict[str, List[str]], text: str,
                    cluster_metadata_manager = None) -> Tuple[List[Dict], List[Dict]]:
        """Main ACO edge inference algorithm."""
        self.text = text
        self.cluster_metadata_manager = cluster_metadata_manager  # Store for use in oracle
        self.initialize_pheromones(clusters)
        
        print(f"🐜 Starting ACO edge inference...")
        print(f"   ⚙️ Max iterations: {self.max_iterations}")
        print(f"   🎯 Samples per iteration: {self.samples_per_iteration}")
        print(f"   💨 Evaporation rate: {self.evaporation_rate}")
        
        for iteration in range(self.max_iterations):
            # Sample edges based on pheromone levels
            sampled_edges = self.sample_edges(iteration)
            
            # Query LLM oracle
            oracle_results = self.query_llm_oracle(sampled_edges)
            
            # Update pheromones
            self.update_pheromones(iteration, oracle_results)
            
            # Check convergence
            if self.check_convergence(iteration):
                print(f"   ✅ Converged at iteration {iteration + 1}")
                break
        
        # Extract final edges
        final_edges = self.extract_final_edges()
        
        print(f"🎯 ACO complete: Found {len(final_edges)} inter-cluster edges")
        print(f"   📊 From {len(self.pheromone_matrix)} possible edges")
        print(f"   💪 Efficiency: {len(final_edges) / len(self.pheromone_matrix) * 100:.1f}% precision")
        
        # Generate intra-cluster edges using standard approach (to avoid circular import)
        intra_cluster_edges = []
        if ENABLE_INTRA_CLUSTER_EDGES:
            print(f"\n🔍 Generating intra-cluster edges using standard approach...")
            # Import here to avoid circular dependency
            from .edge_inference import infer_intra_cluster_edges
            intra_cluster_edges = infer_intra_cluster_edges(
                clusters, text, EDGE_INFERENCE_MODEL, EDGE_INFERENCE_TEMPERATURE, 
                self.cluster_metadata_manager
            )
        
        return final_edges, intra_cluster_edges


# Import required modules
import re 
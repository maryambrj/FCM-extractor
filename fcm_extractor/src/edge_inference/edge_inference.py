import os
import json
from typing import List, Dict, Tuple
from dotenv import load_dotenv
import re
import string
import logging
from datetime import datetime
import itertools
import sys

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))

# Add parent directories to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models import MetaPromptingAgent, get_global_prompt_agent, TaskType
from config.constants import (
    EDGE_INFERENCE_MODEL, EDGE_INFERENCE_TEMPERATURE, EDGE_INFERENCE_BATCH_SIZE,
    CLUSTER_EDGE_BATCH_SIZE, MAX_EDGE_INFERENCE_TEXT_LENGTH,
    DEFAULT_EDGE_INFERENCE_PROMPT, DEFAULT_INTER_CLUSTER_EDGE_PROMPT, DEFAULT_INTRA_CLUSTER_EDGE_PROMPT,
    USE_CONFIDENCE_FILTERING, EDGE_CONFIDENCE_THRESHOLD, ENABLE_INTRA_CLUSTER_EDGES
)
from src.models.llm_client import llm_client

LOG_FILE_PATH = os.path.join(os.getenv('OUTPUT_DIRECTORY', 'fcm_outputs'), 'fcm_edge_inference.log')

def log_to_file(message: str):
    os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True)
    with open(LOG_FILE_PATH, 'a') as f:
        f.write(f"[{datetime.now().isoformat()}] {message}\n")

def log_edge_summary(concept_pairs, llm_lines):
    # Log a summary of LLM responses for edge inference
    summary = []
    for i, (pair, line) in enumerate(zip(concept_pairs, llm_lines)):
        summary.append(f"{pair[0]} -> {pair[1]}: {line.strip()[:60]}")
    log_to_file("Edge inference LLM response summary:\n" + "\n".join(summary))

def normalize_concept_name(name):
    # Lowercase, remove punctuation, and common stopwords
    stopwords = {'the', 'a', 'an', 'of', 'in', 'on', 'at', 'to', 'for', 'with', 'and', 'or'}
    name = name.lower().strip()
    name = name.translate(str.maketrans('', '', string.punctuation))
    tokens = [t for t in name.split() if t not in stopwords]
    return ' '.join(tokens)

def parse_llm_response_for_edges(response: str, pairs: List[Tuple[str, str]], use_confidence: bool) -> List[Dict]:
    """Parse unified format LLM response for edge inference."""
    # Check for empty response first
    if not response.strip():
        print(f"ERROR: Empty LLM response for {len(pairs)} pairs")
        return []
    
    print(f"Raw LLM Response:\n---\n{response[:500]}{'...' if len(response) > 500 else ''}\n---")
    
    # Try LangSmith format first, then fall back to unified format
    langsmith_edges = parse_langsmith_format(response, pairs, use_confidence)
    if langsmith_edges:
        return langsmith_edges
    
    edges = []
    lines = response.strip().split('\n')

    # Parse each line using unified format
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
            if 0 <= pair_num < len(pairs):
                expected_c1, expected_c2 = pairs[pair_num]
                
                # Create edge result
                edge = {
                    "source": concept1,
                    "target": concept2,
                    "weight": 1.0 if relationship == "positive" else -1.0,
                    "relationship": relationship,
                    "confidence": confidence if use_confidence else 1.0,
                    "expected_pair": f"{expected_c1} -> {expected_c2}",
                    "parsed_pair": f"{concept1} -> {concept2}"
                }
                edges.append(edge)
                print(f"✓ Parsed: {concept1} -> {concept2} ({relationship}, conf: {confidence})")
            else:
                print(f"WARNING: Invalid pair number {pair_num + 1} in line: '{line}'")
        else:
            # Check for "no relationship"
            no_match = no_relationship_pattern.match(line)
            if no_match:
                pair_num = int(no_match.group(1)) - 1
                print(f"✓ Parsed: Pair {pair_num + 1} - no relationship")
                # No edge added for "no relationship"
            else:
                print(f"WARNING: Could not parse line: '{line}'")
    
    print(f"Successfully parsed {len(edges)} edges from {len(lines)} lines")
    return edges


def parse_langsmith_format(response: str, pairs: List[Tuple[str, str]], use_confidence: bool) -> List[Dict]:
    """Parse LangSmith format responses for edge inference."""
    edges = []
    
    # Split response into sections for each pair
    sections = re.split(r'Response for Pair (\d+):', response)
    
    if len(sections) < 3:  # No valid sections found
        return []
    
    print(f"Found {(len(sections) - 1) // 2} LangSmith response sections")
    
    for i in range(1, len(sections), 2):
        try:
            pair_num = int(sections[i]) - 1  # Convert to 0-based index
            section_content = sections[i + 1] if i + 1 < len(sections) else ""
            
            if pair_num >= len(pairs):
                print(f"WARNING: Invalid pair number {pair_num + 1}")
                continue
            
            # Parse the section content
            relationship_match = re.search(r'- Relationship:\s*(Yes|No)', section_content, re.IGNORECASE)
            direction_match = re.search(r'- Direction:\s*([AB])\s*->\s*([AB])', section_content, re.IGNORECASE)
            polarity_match = re.search(r'- Polarity:\s*(Negative|Positive|Mixed)', section_content, re.IGNORECASE)
            confidence_match = re.search(r'- Confidence:\s*([0-1]\.\d+)', section_content)
            
            if not relationship_match or relationship_match.group(1).lower() == 'no':
                print(f"✓ Parsed: Pair {pair_num + 1} - no relationship")
                continue
            
            if not direction_match or not polarity_match or not confidence_match:
                print(f"WARNING: Incomplete data for pair {pair_num + 1}")
                continue
            
            # Get the concepts for this pair
            expected_c1, expected_c2 = pairs[pair_num]
            
            # Determine source and target based on direction
            direction_from = direction_match.group(1).upper()
            direction_to = direction_match.group(2).upper()
            
            if direction_from == 'A' and direction_to == 'B':
                source, target = expected_c1, expected_c2
            elif direction_from == 'B' and direction_to == 'A':
                source, target = expected_c2, expected_c1
            else:
                print(f"WARNING: Invalid direction for pair {pair_num + 1}: {direction_from} -> {direction_to}")
                continue
            
            # Determine polarity
            polarity = polarity_match.group(1).lower()
            if polarity == "mixed":
                # For mixed, we'll default to positive but could handle differently
                weight = 1.0
                relationship = "positive"
            else:
                weight = 1.0 if polarity == "positive" else -1.0
                relationship = polarity
            
            confidence = float(confidence_match.group(1))
            
            # Create edge result
            edge = {
                "source": source,
                "target": target,
                "weight": weight,
                "relationship": relationship,
                "confidence": confidence if use_confidence else 1.0,
                "expected_pair": f"{expected_c1} -> {expected_c2}",
                "parsed_pair": f"{source} -> {target}",
                "format": "langsmith"
            }
            edges.append(edge)
            print(f"✓ Parsed LangSmith: {source} -> {target} ({relationship}, conf: {confidence})")
            
        except (ValueError, IndexError) as e:
            print(f"WARNING: Error parsing pair {i//2 + 1}: {e}")
            continue
    
    print(f"Successfully parsed {len(edges)} edges from LangSmith format")
    return edges

# Maximum safe text length for edge inference - imported from constants.py
# (was previously hardcoded as 2000, now configurable)

def extract_relevant_text_for_concepts(text: str, concept_pairs: List[Tuple[str, str]], max_length: int = MAX_EDGE_INFERENCE_TEXT_LENGTH) -> str:
    """
    Extract the most relevant portions of text for concept analysis.
    Optimized for cost efficiency with larger batches.
    """
    
    if len(text) <= max_length:
        return text
    
    # Collect all unique concepts from pairs
    all_concepts = set()
    for c1, c2 in concept_pairs:
        all_concepts.add(c1.lower())
        all_concepts.add(c2.lower())
    
    # Split text into sentences for better granularity
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    
    # Score sentences by concept relevance
    scored_sentences = []
    for i, sentence in enumerate(sentences):
        sentence_lower = sentence.lower()
        
        # Count concept matches in this sentence
        concept_matches = sum(1 for concept in all_concepts if concept in sentence_lower)
        
        # Bonus for sentences with multiple concepts (potential relationships)
        relationship_bonus = concept_matches * 2 if concept_matches > 1 else concept_matches
        
        # Bonus for causal language
        causal_words = ['because', 'causes', 'leads to', 'results in', 'due to', 'affects', 'influences', 'impacts']
        causal_bonus = sum(1 for word in causal_words if word in sentence_lower)
        
        total_score = relationship_bonus + causal_bonus
        
        if total_score > 0:  # Only include sentences with relevance
            scored_sentences.append((total_score, i, sentence))
    
    # Sort by score (highest first) and select best sentences
    scored_sentences.sort(key=lambda x: x[0], reverse=True)
    
    # Build result with best sentences up to max_length
    selected_text = ""
    selected_indices = set()
    
    for score, idx, sentence in scored_sentences:
        candidate_text = selected_text + (" " if selected_text else "") + sentence + "."
        if len(candidate_text) <= max_length:
            selected_text = candidate_text
            selected_indices.add(idx)
        else:
            break
    
    # If we have room, add context sentences (before/after selected ones)
    for score, idx, sentence in scored_sentences:
        if idx in selected_indices:
            continue
            
        # Check adjacent sentences for context
        for adj_idx in [idx-1, idx+1]:
            if (adj_idx >= 0 and adj_idx < len(sentences) and 
                adj_idx not in selected_indices):
                
                context_sentence = sentences[adj_idx]
                candidate_text = selected_text + " " + context_sentence + "."
                
                if len(candidate_text) <= max_length:
                    selected_text = candidate_text
                    selected_indices.add(adj_idx)
                    break
    
    return selected_text if selected_text else text[:max_length]

def batch_llm_edge_queries(concept_pairs: List[Tuple[str, str]], text: str, 
                             model: str = EDGE_INFERENCE_MODEL, 
                             temperature: float = EDGE_INFERENCE_TEMPERATURE,
                             _is_fallback: bool = False,
                             prompt_template: str = None,
                             use_dynamic_prompting: bool = True) -> List[Dict]:
    """Process concept pairs in batches to infer edges using LLM with unified prompt format."""
    
    if not concept_pairs:
        return []
    
    # Extract relevant text if the full text is too long
    if len(text) > MAX_EDGE_INFERENCE_TEXT_LENGTH:
        print(f"  Text too long for edge inference ({len(text)} chars), extracting relevant portions...")
        relevant_text = extract_relevant_text_for_concepts(text, concept_pairs)
        print(f"  Using {len(relevant_text)} characters of relevant text")
    else:
        relevant_text = text
    
    # Validate that concepts actually exist in the text
    valid_pairs = []
    skipped_pairs = []
    
    for c1, c2 in concept_pairs:
        c1_in_text = c1.lower() in relevant_text.lower()
        c2_in_text = c2.lower() in relevant_text.lower()
        
        if c1_in_text and c2_in_text:
            valid_pairs.append((c1, c2))
        else:
            skipped_pairs.append((c1, c2))
            print(f"    Skipping pair '{c1}' -> '{c2}': concepts not found in text")
    
    if not valid_pairs:
        print(f"    No valid concept pairs found in text, skipping LLM call")
        return []
    
    if skipped_pairs:
        print(f"    Processing {len(valid_pairs)}/{len(concept_pairs)} pairs (skipped {len(skipped_pairs)} with missing concepts)")
    
    # Generate dynamic prompt or use template
    if use_dynamic_prompting and prompt_template is None:
        # Use dynamic prompt agent for contextually-aware edge inference
        dynamic_agent = get_global_prompt_agent()
        
        # Create context for edge inference
        concept_names = [c1 for c1, c2 in valid_pairs] + [c2 for c1, c2 in valid_pairs]
        
        base_prompt, generation_metadata = dynamic_agent.generate_dynamic_prompt(
            TaskType.EDGE_INFERENCE, 
            relevant_text,
            concept_pairs=valid_pairs,
            concept_names=concept_names
        )
        
        # Format the dynamic prompt with specific pairs
        pairs_text = ""
        for i, (c1, c2) in enumerate(valid_pairs):
            pairs_text += f"Pair {i+1}: '{c1}' and '{c2}'\n"
        
        prompt = f"{base_prompt}\n\nText to analyze: {relevant_text}\n\nConcept pairs to analyze:\n{pairs_text}"
        
    else:
        # Use provided template or fallback to default
        if prompt_template is None:
            prompt_template = DEFAULT_INTRA_CLUSTER_EDGE_PROMPT
            
        pairs_text = ""
        for i, (c1, c2) in enumerate(valid_pairs):
            pairs_text += f"Pair {i+1}: '{c1}' and '{c2}'\n"
        
        prompt = prompt_template.format(
            text=relevant_text,
            pairs=pairs_text
        )

    messages = [
        {"role": "system", "content": "You are an expert at analyzing interview transcripts for causal relationships between concepts."},
        {"role": "user", "content": prompt}
    ]
    
    try:
        # Debug: show prompt size breakdown
        total_prompt_length = len(messages[0]["content"]) + len(messages[1]["content"])
        print(f"    Batch size: {len(valid_pairs)} pairs, Text: {len(relevant_text)} chars, Total prompt: {total_prompt_length} chars")
        
        # Get LLM response
        content, _ = llm_client.chat_completion(model, messages, temperature)
        
        print(f"--- RAW LLM INTRA-CLUSTER RESPONSE ---\n{content}\n------------------------------------")

        # Check for empty response
        if not content.strip():
            print(f"  ⚠️ Warning: LLM returned empty response for {len(valid_pairs)} concept pairs")
            print(f"  Text: {len(relevant_text)} chars, Total prompt: {total_prompt_length} chars")
            
            # Try processing pairs individually as fallback (only if not already in fallback mode)
            if len(concept_pairs) > 1 and not _is_fallback:
                print(f"    Trying individual pair processing as fallback...")
                individual_results = []
                for i, (c1, c2) in enumerate(concept_pairs[:2]):  # Limit to first 2 to avoid too many calls
                    print(f"      Processing pair {i+1}: '{c1}' -> '{c2}'")
                    try:
                        single_result = batch_llm_edge_queries([(c1, c2)], relevant_text, model, temperature, _is_fallback=True)
                        individual_results.extend(single_result)
                        if single_result:
                            print(f"        ✅ Success: {len(single_result)} edge(s)")
                        else:
                            print(f"        ❌ Failed: empty response")
                    except Exception as e:
                        print(f"        ❌ Error: {e}")
                return individual_results
            return []
        
        # Parse response using unified parser
        edges = parse_llm_response_for_edges(content, valid_pairs, use_confidence=True)
        
        # Add type information for intra-cluster edges
        for edge in edges:
            edge['type'] = 'intra_cluster'
        
        return edges
        
    except Exception as e:
        print(f"  ❌ Error in edge inference LLM call: {e}")
        return []

def llm_edge_query(source: str, target: str, text: str, model: str = EDGE_INFERENCE_MODEL, 
                   temperature: float = EDGE_INFERENCE_TEMPERATURE) -> Tuple[str, float]:
    """Single edge query (for backward compatibility)."""
    results = batch_llm_edge_queries([(source, target)], text, model, temperature)
    return results[0]

def infer_intra_cluster_edges(clusters: Dict[str, List[str]], text: str,
                              model: str = EDGE_INFERENCE_MODEL,
                              temperature: float = EDGE_INFERENCE_TEMPERATURE,
                              cluster_metadata_manager = None) -> List[Dict]:
    """
    Infer edges within individual clusters (between concepts in the same cluster).
    
    Args:
        clusters: Dictionary mapping cluster names to concept lists
        text: Full input text (or will use relevant contexts from metadata)
        model: LLM model to use
        temperature: Temperature for LLM
        cluster_metadata_manager: ClusterMetadataManager with concept contexts (optional)
    
    Returns:
        List of intra-cluster edge dictionaries
    """
    if not ENABLE_INTRA_CLUSTER_EDGES:
        print("Intra-cluster edge inference disabled in configuration.")
        return []
    
    all_intra_edges = []
    
    for cluster_name, concepts in clusters.items():
        if len(concepts) < 2:
            continue  # Need at least 2 concepts to form edges
        
        # Generate all possible pairs within this cluster
        from itertools import combinations
        concept_pairs = list(combinations(concepts, 2))
        
        if not concept_pairs:
            continue
        
        print(f"  Inferring intra-cluster edges for '{cluster_name}' ({len(concept_pairs)} pairs)...")
        
        # Extract relevant text for this cluster's concepts
        relevant_text = text  # Default to full text
        
        if cluster_metadata_manager is not None:
            # Get contexts for this cluster
            cluster_contexts = []
            if cluster_name in cluster_metadata_manager.clusters:
                cluster_contexts = cluster_metadata_manager.clusters[cluster_name].get_all_contexts()
            
            if cluster_contexts:
                relevant_text = "\n\n".join(cluster_contexts)
                print(f"    Using {len(cluster_contexts)} relevant contexts ({len(relevant_text)} chars)")
        
        # Process pairs in batches
        for i in range(0, len(concept_pairs), EDGE_INFERENCE_BATCH_SIZE):
            batch_pairs = concept_pairs[i:i + EDGE_INFERENCE_BATCH_SIZE]
            
            # Use the intra-cluster prompt template
            batch_edges = batch_llm_edge_queries(
                batch_pairs, relevant_text, model, temperature, 
                prompt_template=DEFAULT_INTRA_CLUSTER_EDGE_PROMPT
            )
            
            # Add cluster information and type to each edge
            for edge in batch_edges:
                edge['cluster'] = cluster_name
                edge['type'] = 'intra_cluster'
            
            all_intra_edges.extend(batch_edges)
    
    print(f"Intra-cluster edge inference complete: Found {len(all_intra_edges)} intra-cluster edges.")
    return all_intra_edges


def infer_cluster_edge_grounded(cluster_a_name: str, cluster_a_concepts: List[str],
                               cluster_b_name: str, cluster_b_concepts: List[str],
                               text: str, model: str, temperature: float) -> Dict:
    """
    Infer edges between clusters by showing the LLM the actual concepts, not abstract names.
    This follows the principle that LLMs can't reason about invented cluster names.
    """
    
    # Extract relevant text for all concepts from both clusters
    all_concepts = cluster_a_concepts + cluster_b_concepts
    mock_pairs = [(all_concepts[0], all_concepts[-1])]  # Just for text extraction
    
    if len(text) > MAX_EDGE_INFERENCE_TEXT_LENGTH:
        relevant_text = extract_relevant_text_for_concepts(text, mock_pairs)
    else:
        relevant_text = text
    
    # Check if concepts from both clusters are present in the text
    cluster_a_in_text = any(concept.lower() in relevant_text.lower() for concept in cluster_a_concepts)
    cluster_b_in_text = any(concept.lower() in relevant_text.lower() for concept in cluster_b_concepts)
    
    if not (cluster_a_in_text and cluster_b_in_text):
        print(f"    Skipping cluster pair '{cluster_a_name}' -> '{cluster_b_name}': concepts not found in text")
        return None
    
    # Create concept-grounded prompt
    cluster_a_list = ", ".join(cluster_a_concepts)
    cluster_b_list = ", ".join(cluster_b_concepts)
    
    prompt = f"""Interview text: "{relevant_text}"

Based on this interview, analyze the relationship between these two concept groups:

Cluster A ({cluster_a_name}): [{cluster_a_list}]
Cluster B ({cluster_b_name}): [{cluster_b_list}]

Question: Do the interview participants discuss any causal relationships between concepts in Cluster A and concepts in Cluster B?

Look for evidence where:
- Concepts from Cluster A influence concepts from Cluster B, OR
- Concepts from Cluster B influence concepts from Cluster A

Respond with one of:
"A causes B positive" - Cluster A concepts increase Cluster B concepts
"A causes B negative" - Cluster A concepts decrease Cluster B concepts  
"B causes A positive" - Cluster B concepts increase Cluster A concepts
"B causes A negative" - Cluster B concepts decrease Cluster A concepts
"no relationship" - No clear causal relationship discussed

Response:"""

    messages = [
        {"role": "system", "content": "Analyze interview transcripts for causal relationships between concept groups. Focus on evidence from the actual conversation."},
        {"role": "user", "content": prompt}
    ]
    
    try:
        content, _ = llm_client.chat_completion(model, messages, temperature)
        
        if not content.strip():
            return None
        
        # Parse the response
        response_lower = content.strip().lower()
        
        if 'no relationship' in response_lower:
            return None
        elif 'a causes b' in response_lower:
            source, target = cluster_a_name, cluster_b_name
        elif 'b causes a' in response_lower:
            source, target = cluster_b_name, cluster_a_name
        else:
            print(f"    Warning: Could not parse cluster edge response: '{content.strip()}'")
            return None
        
        # Determine weight
        weight = 1  # Default to positive
        if 'negative' in response_lower:
            weight = -1
        
        return {
            'source': source,
            'target': target,
            'weight': weight,
            'confidence': 0.8,
            'type': 'inter_cluster'
        }
        
    except Exception as e:
        print(f"    Error in cluster edge inference: {e}")
        return None

def parse_cluster_edge_response(response: str, cluster_pairs_data: List[Tuple[str, List[str], str, List[str]]]) -> List[Dict]:
    """
    Parse the structured LLM response for inter-cluster edge inference,
    using confidence scores as a soft threshold.
    """
    edges = []
    
    # Split the response into lines and process each line
    lines = response.strip().split('\n')
    
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
            if 0 <= pair_num < len(cluster_pairs_data):
                cluster_a_name, cluster_a_concepts, cluster_b_name, cluster_b_concepts = cluster_pairs_data[pair_num]
                
                # Determine weight based on relationship
                weight = 1.0 if relationship == "positive" else -1.0
                
                # Use confidence as the primary filter only if confidence filtering is enabled
                if USE_CONFIDENCE_FILTERING and confidence < EDGE_CONFIDENCE_THRESHOLD:
                    continue  # Skip if below threshold and filtering is enabled
                
                # Create edge result
                edge = {
                    'source': concept1,
                    'target': concept2,
                    'weight': weight,
                    'confidence': confidence,
                    'type': 'inter_cluster',
                    'rationale': f"Direct causal relationship: {concept1} {relationship}ly affects {concept2}",
                    'cluster_a': cluster_a_name,
                    'cluster_b': cluster_b_name
                }
                edges.append(edge)
                
        elif no_relationship_pattern.match(line):
            # Skip "no relationship" responses
            continue
        else:
            # Try to parse other formats or log unparseable lines
            print(f"    Warning: Could not parse line: {line[:100]}...")
            continue
            
    print(f"    Parsed {len(edges)} edges from {len(lines)} response lines")
    return edges


def batch_cluster_edge_inference(cluster_pairs_data: List[Tuple[str, List[str], str, List[str]]], 
                                text: str, model: str, temperature: float,
                                prompt_template: str = None,
                                cluster_metadata_manager = None,
                                use_dynamic_prompting: bool = True) -> List[Dict]:
    """
    Infer inter-cluster edges using a batched approach with a structured prompt.
    Uses only relevant text portions from cluster metadata when available.
    """
    if not cluster_pairs_data:
        return []
    
    # Extract relevant text from metadata if available
    relevant_text = text  # Default to full text
    
    if cluster_metadata_manager is not None:
        # Collect all relevant contexts from cluster metadata
        all_contexts = []
        seen_contexts = set()  # To avoid duplicates
        
        for cluster_a_name, _, cluster_b_name, _ in cluster_pairs_data:
            # Get contexts for this cluster pair
            contexts = cluster_metadata_manager.get_cluster_contexts_for_edge_inference(
                cluster_a_name, cluster_b_name
            )
            
            # Add unique contexts
            for context in contexts:
                if context not in seen_contexts:
                    seen_contexts.add(context)
                    all_contexts.append(context)
        
        if all_contexts:
            # Join contexts with clear separation
            relevant_text = "\n\n".join(all_contexts)
            print(f"    Using {len(all_contexts)} relevant text contexts ({len(relevant_text)} chars) instead of full text ({len(text)} chars)")
        else:
            print(f"    Warning: No relevant contexts found in metadata, using full text")
        
    # Generate dynamic prompt or use template
    if use_dynamic_prompting and prompt_template is None:
        # Use dynamic prompt agent for inter-cluster edge inference
        dynamic_agent = get_global_prompt_agent()
        
        # Extract cluster information for context
        cluster_info = []
        all_concepts = []
        for cluster_a_name, cluster_a_concepts, cluster_b_name, cluster_b_concepts in cluster_pairs_data:
            cluster_info.append({
                'cluster_a': cluster_a_name,
                'concepts_a': cluster_a_concepts[:3],
                'cluster_b': cluster_b_name, 
                'concepts_b': cluster_b_concepts[:3]
            })
            all_concepts.extend(cluster_a_concepts[:3])
            all_concepts.extend(cluster_b_concepts[:3])
        
        base_prompt, generation_metadata = dynamic_agent.generate_dynamic_prompt(
            TaskType.EDGE_INFERENCE,
            relevant_text,
            cluster_pairs=cluster_info,
            all_concepts=list(set(all_concepts)),
            inference_type="inter_cluster"
        )
        
        # Format cluster pairs information
        pairs_text = ""
        for i, (cluster_a_name, cluster_a_concepts, cluster_b_name, cluster_b_concepts) in enumerate(cluster_pairs_data):
            pairs_text += (
                f"\n--- Pair {i+1} ---\n"
                f"Cluster A: \"{cluster_a_name}\" (Concepts: {', '.join(cluster_a_concepts[:3])})\n"
                f"Cluster B: \"{cluster_b_name}\" (Concepts: {', '.join(cluster_b_concepts[:3])})\n"
            )
        
        prompt = f"{base_prompt}\n\nText to analyze: {relevant_text}\n\nCluster pairs to analyze:{pairs_text}"
        
    else:
        # Use provided template or fallback to default
        if prompt_template is None:
            prompt_template = DEFAULT_INTER_CLUSTER_EDGE_PROMPT
            
        # Format the pairs for the prompt
        pairs_text = ""
        for i, (cluster_a_name, cluster_a_concepts, cluster_b_name, cluster_b_concepts) in enumerate(cluster_pairs_data):
            pairs_text += (
                f"\n--- Pair {i+1} ---\n"
                f"Cluster A: \"{cluster_a_name}\" (Concepts: {', '.join(cluster_a_concepts[:3])})\n"
                f"Cluster B: \"{cluster_b_name}\" (Concepts: {', '.join(cluster_b_concepts[:3])})\n"
            )

        # Create the full prompt with relevant text only
        prompt = prompt_template.format(text=relevant_text, pairs=pairs_text)
    
    messages = [{"role": "user", "content": prompt}]
    
    try:
        content, _ = llm_client.chat_completion(model, messages, temperature, max_tokens=4096)
        
        if not content.strip():
            print("    Warning: Received empty response from LLM for inter-cluster batch.")
            return []
        
        # DEBUG: Print the LLM response to help with troubleshooting
        # print(f"\n=== DEBUG: LLM Response for Inter-cluster Edge Inference ===")
        # print(f"Response length: {len(content)} characters")
        # print(f"First 500 characters: {content[:500]}...")
        # print("=" * 60)
            
        # The new parsing function will handle the structured response
        return parse_cluster_edge_response(content, cluster_pairs_data)
        
    except Exception as e:
        print(f"    Error during inter-cluster edge inference: {e}")
        return []

def infer_edges(clusters: Dict[str, List[str]], text: str, 
                model: str = EDGE_INFERENCE_MODEL, 
                temperature: float = EDGE_INFERENCE_TEMPERATURE,
                cluster_metadata_manager = None) -> Tuple[List[Dict], List[Dict]]:
    """
    Infer causal edges between clusters using a concept-grounded approach with efficient batching.
    This function now focuses exclusively on inter-cluster edges.
    
    Args:
        clusters: Dictionary mapping cluster names to concept lists
        text: Full input text
        model: LLM model to use
        temperature: Temperature for LLM
        cluster_metadata_manager: ClusterMetadataManager with concept contexts (optional)
    """
    if len(clusters) < 2:
        return [], []
    
    # Get cluster pairs for inter-cluster edge inference
    cluster_names = list(clusters.keys())
    cluster_pairs = list(itertools.combinations(cluster_names, 2))
    
    print(f"Processing edge inference for {len(cluster_pairs)} cluster pairs...")
    
    # Prepare cluster pairs data for batch processing
    cluster_pairs_data = []
    for cluster_a_name, cluster_b_name in cluster_pairs:
        cluster_a_concepts = clusters[cluster_a_name]
        cluster_b_concepts = clusters[cluster_b_name]
        cluster_pairs_data.append((cluster_a_name, cluster_a_concepts, cluster_b_name, cluster_b_concepts))
    
    all_inter_cluster_edges = []
    # Process the pairs in batches
    for i in range(0, len(cluster_pairs_data), CLUSTER_EDGE_BATCH_SIZE):
        batch = cluster_pairs_data[i:i + CLUSTER_EDGE_BATCH_SIZE]
        
        # Use the new specialized prompt for inter-cluster edges
        inter_cluster_edges = batch_cluster_edge_inference(
            batch, text, model, temperature, 
            prompt_template=DEFAULT_INTER_CLUSTER_EDGE_PROMPT,
            cluster_metadata_manager=cluster_metadata_manager
        )
        all_inter_cluster_edges.extend(inter_cluster_edges)

    # Infer intra-cluster edges
    print(f"\nInferring intra-cluster edges...")
    intra_cluster_edges = infer_intra_cluster_edges(
        clusters, text, model, temperature, cluster_metadata_manager
    )
    
    print(f"\nEdge inference complete: Found {len(all_inter_cluster_edges)} inter-cluster edges and {len(intra_cluster_edges)} intra-cluster edges.")
    return all_inter_cluster_edges, intra_cluster_edges

def infer_edges_original(clusters: Dict[str, List[str]], text: str, 
                        model: str = EDGE_INFERENCE_MODEL, 
                        temperature: float = EDGE_INFERENCE_TEMPERATURE) -> Tuple[List[Dict], List[Dict]]:
    """
    Original edge inference approach (kept for comparison/fallback).
    Infer causal edges at both the concept and cluster level using the full text context.
    """
    
    # 1. Get all original concepts and create pairs
    all_original_concepts = [concept for concept_list in clusters.values() for concept in concept_list]
    if len(all_original_concepts) < 2:
        return [], []
    concept_pairs = list(itertools.combinations(all_original_concepts, 2))
    
    # Create a reverse mapping from original concept to its cluster name
    concept_to_cluster_map = {concept: cluster_name for cluster_name, concepts in clusters.items() for concept in concepts}
    
    all_concept_edges = []
    
    # 2. Process all pairs in batches against the full text
    print(f"Processing edge inference for {len(concept_pairs)} original concept pairs...")
    for batch_idx in range(0, len(concept_pairs), EDGE_INFERENCE_BATCH_SIZE):
        batch_pairs = concept_pairs[batch_idx:batch_idx + EDGE_INFERENCE_BATCH_SIZE]
        batch_results = batch_llm_edge_queries(batch_pairs, text, model, temperature, prompt_template=DEFAULT_EDGE_INFERENCE_PROMPT)
        all_concept_edges.extend(batch_results)

    # 3. Separate edges into inter-cluster and intra-cluster
    inter_cluster_edges = []
    intra_cluster_edges = []
    
    for edge in all_concept_edges:
        source_concept = edge['source']
        target_concept = edge['target']
        
        source_cluster = concept_to_cluster_map.get(source_concept)
        target_cluster = concept_to_cluster_map.get(target_concept)
        
        if source_cluster and target_cluster:
            if source_cluster == target_cluster:
                intra_cluster_edges.append(edge)
            else:
                inter_cluster_edges.append(edge)

    # 4. Aggregate the inter-cluster edges up to the cluster level
    cluster_edge_weights = {}
    for edge in inter_cluster_edges:
        source_cluster = concept_to_cluster_map.get(edge['source'])
        target_cluster = concept_to_cluster_map.get(edge['target'])
        
        # Skip edges with weight 0 (neither relationships) in aggregation
        if edge['weight'] == 0:
            continue
            
        if source_cluster and target_cluster and source_cluster != target_cluster:
            key = tuple(sorted((source_cluster, target_cluster)))
            if key not in cluster_edge_weights:
                cluster_edge_weights[key] = []
            
            # Store weight, considering direction
            if (source_cluster, target_cluster) == key:
                cluster_edge_weights[key].append(edge['weight'])
            else:
                cluster_edge_weights[key].append(-edge['weight']) # Reverse weight for direction
    
    # 5. Create final aggregated cluster edges
    final_cluster_edges = []
    print(f"Aggregating {len(cluster_edge_weights)} cluster pairs...")
    for (c1, c2), weights in cluster_edge_weights.items():
        if not weights:
            continue
        
        avg_weight = sum(weights) / len(weights)
        print(f"  {c1} <-> {c2}: {len(weights)} edges, avg_weight={avg_weight:.3f}")
        
        # Determine final direction and weight
        if abs(avg_weight) > 0.01: # Threshold to create an edge (lowered from 0.05 to be more permissive)
            final_weight = 1 if avg_weight > 0 else -1
            if avg_weight > 0:
                final_cluster_edges.append({'source': c1, 'target': c2, 'weight': final_weight, 'confidence': abs(avg_weight)})
            else:
                final_cluster_edges.append({'source': c2, 'target': c1, 'weight': final_weight, 'confidence': abs(avg_weight)})
            print(f"    -> Created edge: {final_cluster_edges[-1]}")
        else:
            print(f"    -> Filtered out (below threshold)")
    
    print(f"Final cluster edges: {len(final_cluster_edges)}")
    print(f"Intra-cluster edges: {len(intra_cluster_edges)}")

    return final_cluster_edges, intra_cluster_edges

if __name__ == "__main__":
    clusters = {0: ["social isolation"], 1: ["depression"]}
    text = "Social isolation leads to poor sleep and depression."
    edges = infer_edges(clusters, text)
    print("Inferred edges:", edges) 
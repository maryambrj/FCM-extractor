import os
import json
import time
from typing import List, Dict, Tuple
from dotenv import load_dotenv
import re
import sys

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))

# Add parent directories to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models import MetaPromptingAgent, get_global_prompt_agent, TaskType
from config.constants import (
    CONCEPT_EXTRACTION_MODEL, CONCEPT_EXTRACTION_TEMPERATURE,
    CONCEPT_EXTRACTION_N_PROMPTS
)
from src.models.llm_client import llm_client
from src.models.cluster_metadata import ConceptMetadata

# Maximum safe text length for the model (found through testing)
MAX_TEXT_LENGTH = 8000

def chunk_text(text: str, max_length: int = MAX_TEXT_LENGTH, overlap: int = 500) -> List[str]:
    """Split text into overlapping chunks to handle long documents."""
    if len(text) <= max_length:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        # Calculate end position
        end = start + max_length
        
        # If this is not the last chunk, try to break at a sentence boundary
        if end < len(text):
            # Look for sentence endings within the last 1000 characters
            sentence_ends = ['.', '!', '?', '\n\n']
            best_break = end
            
            for i in range(end - 1000, end):
                if i > 0 and text[i] in sentence_ends:
                    best_break = i + 1
            
            end = best_break
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # Move start position with overlap
        start = end - overlap
        if start >= len(text):
            break
    
    return chunks

def llm_extract_concepts(text: str, model: str = CONCEPT_EXTRACTION_MODEL, 
                         temperature: float = CONCEPT_EXTRACTION_TEMPERATURE, meta_agent: MetaPromptingAgent = None) -> List[str]:
    """Call LLM API to extract concepts from text using meta-prompting."""
    
    # Check if text is too long and needs chunking
    if len(text) > MAX_TEXT_LENGTH:
        print(f"  Text too long ({len(text)} chars), using chunking strategy...")
        chunks = chunk_text(text)
        print(f"  Split into {len(chunks)} chunks")
        
        all_concepts = []
        for i, chunk in enumerate(chunks):
            print(f"    Processing chunk {i+1}/{len(chunks)} ({len(chunk)} chars)...")
            chunk_concepts = llm_extract_concepts(chunk, model, temperature, meta_agent)
            all_concepts.extend(chunk_concepts)
            print(f"    Chunk {i+1} extracted {len(chunk_concepts)} concepts")
        
        # Deduplicate across chunks
        unique_concepts = list(set(all_concepts))
        print(f"  Combined {len(all_concepts)} concepts into {len(unique_concepts)} unique concepts")
        return unique_concepts
    
    # Use dynamic prompt agent for contextually-aware prompt generation
    dynamic_agent = get_global_prompt_agent()
    start_time = time.time()
    adaptive_prompt, generation_metadata = dynamic_agent.generate_dynamic_prompt(
        TaskType.CONCEPT_EXTRACTION, text
    )
    
    messages = [
        {"role": "system", "content": "Extract key concepts or entities from the text as short terms (1-3 words each). Return ONLY a comma-separated list with no bullets, numbers, asterisks, markdown, explanations, or long phrases."},
        {"role": "user", "content": f"{adaptive_prompt}\n\n{text}"}
    ]
    
    try:
        content, _ = llm_client.chat_completion(model, messages, temperature)
        execution_time = time.time() - start_time
        
        # Check if we got an empty response (indicates potential failure)
        if not content.strip():
            print(f"    ⚠️ Warning: LLM returned empty response for text of length {len(text)}")
            
            # Record poor performance
            if 'text_analysis' in generation_metadata:
                dynamic_agent.record_performance(
                    prompt=adaptive_prompt,
                    task_type=TaskType.CONCEPT_EXTRACTION,
                    text_analysis=generation_metadata['text_analysis'],
                    performance_score=0.0,
                    execution_time=execution_time,
                    success=False,
                    error_message="Empty LLM response"
                )
            return []
        
    except Exception as e:
        print(f"    ❌ Error in LLM call: {e}")
        return []
    
    # Clean and split the response
    concepts = []
    for line in content.split('\n'):
        line = line.strip()
        # Remove prefixes, markdown, and numbers in parentheses
        line = re.sub(r'^[-•*0-9. ]+|\*\*|:|\(.*\)$', '', line).strip()
        if line:
            # Trim to max 3 words
            concepts.append(' '.join(line.split()[:3]))
    
    # If no lines, try comma separation with same cleaning
    if len(concepts) <= 1:
        concepts = []
        for c in content.split(','):
            c = re.sub(r'^[-•*0-9. ]+|\*\*|:|\(.*\)$', '', c.strip())
            if c:
                concepts.append(' '.join(c.split()[:3]))
    
    # Record successful performance
    if concepts and 'text_analysis' in generation_metadata:
        performance_score = min(len(concepts) / 10.0, 1.0)  # Score based on concept count
        dynamic_agent.record_performance(
            prompt=adaptive_prompt,
            task_type=TaskType.CONCEPT_EXTRACTION,
            text_analysis=generation_metadata['text_analysis'],
            performance_score=performance_score,
            execution_time=execution_time,
            success=True
        )
    
    return concepts

def llm_extract_concepts_with_context(text: str, chunk_index: int = 0, 
                                     model: str = CONCEPT_EXTRACTION_MODEL, 
                                     temperature: float = CONCEPT_EXTRACTION_TEMPERATURE, 
                                     meta_agent: MetaPromptingAgent = None) -> Tuple[List[str], Dict[str, ConceptMetadata]]:
    """
    Extract concepts and track their source contexts.
    
    Returns:
        Tuple of (concept_list, concept_metadata_dict)
    """
    
    # Check if text is too long and needs chunking
    if len(text) > MAX_TEXT_LENGTH:
        print(f"  Text too long ({len(text)} chars), using chunking strategy...")
        chunks = chunk_text(text)
        print(f"  Split into {len(chunks)} chunks")
        
        all_concepts = []
        all_concept_metadata = {}
        
        for i, chunk in enumerate(chunks):
            print(f"    Processing chunk {i+1}/{len(chunks)} ({len(chunk)} chars)...")
            chunk_concepts, chunk_metadata = llm_extract_concepts_with_context(
                chunk, chunk_index=i, model=model, temperature=temperature, meta_agent=meta_agent
            )
            
            # Merge concepts and metadata
            for concept in chunk_concepts:
                if concept not in all_concept_metadata:
                    all_concept_metadata[concept] = ConceptMetadata(
                        concept=concept,
                        source_contexts=[],
                        chunk_indices=[]
                    )
                
                # Add metadata from this chunk
                if concept in chunk_metadata:
                    chunk_meta = chunk_metadata[concept]
                    all_concept_metadata[concept].source_contexts.extend(chunk_meta.source_contexts)
                    all_concept_metadata[concept].chunk_indices.extend(chunk_meta.chunk_indices)
            
            all_concepts.extend(chunk_concepts)
            print(f"    Chunk {i+1} extracted {len(chunk_concepts)} concepts")
        
        # Deduplicate concepts while preserving metadata
        unique_concepts = list(set(all_concepts))
        print(f"  Combined {len(all_concepts)} concepts into {len(unique_concepts)} unique concepts")
        
        return unique_concepts, all_concept_metadata
    
    # Extract concepts from single chunk
    concepts = llm_extract_concepts(text, model, temperature, meta_agent)
    
    # Create context metadata for each concept
    concept_metadata = {}
    
    # Extract sentences/contexts where each concept appears
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    
    for concept in concepts:
        concept_lower = concept.lower()
        matching_contexts = []
        
        # Find sentences containing this concept
        for sentence in sentences:
            if concept_lower in sentence.lower():
                # Clean up the sentence
                clean_sentence = sentence.strip()
                if len(clean_sentence) > 20:  # Skip very short fragments
                    matching_contexts.append(clean_sentence)
        
        # If no sentence matches, look for the concept in context
        if not matching_contexts:
            # Try to find the concept in the text and extract surrounding context
            text_lower = text.lower()
            concept_pos = text_lower.find(concept_lower)
            if concept_pos >= 0:
                # Extract context around the concept (±200 chars)
                start = max(0, concept_pos - 200)
                end = min(len(text), concept_pos + len(concept) + 200)
                context = text[start:end].strip()
                if context:
                    matching_contexts.append(f"...{context}...")
        
        concept_metadata[concept] = ConceptMetadata(
            concept=concept,
            source_contexts=matching_contexts,
            chunk_indices=[chunk_index]
        )
    
    return concepts, concept_metadata

def normalize_concept(concept: str) -> str:
    """Normalize concept names for deduplication."""
    return concept.lower().strip()

def extract_concepts(text: str, n_prompts: int = CONCEPT_EXTRACTION_N_PROMPTS, model: str = CONCEPT_EXTRACTION_MODEL, 
                    temperature: float = CONCEPT_EXTRACTION_TEMPERATURE) -> List[str]:
    """Extract concepts from the full text using meta-prompting and deduplicate results."""
    all_concepts = []
    
    # Initialize meta-prompting agent
    meta_agent = MetaPromptingAgent(model=model, temperature=temperature)
    
    print(f"Extracting concepts from full text ({len(text)} chars) with {n_prompts} prompt(s)...")
    
    # Show chunking info if text is long
    if len(text) > MAX_TEXT_LENGTH:
        estimated_chunks = len(chunk_text(text))
        print(f"Text is long, will use chunking strategy ({estimated_chunks} estimated chunks)")
    
    # Extract concepts from the full text multiple times
    for i in range(n_prompts):
        concepts = llm_extract_concepts(text, model=model, temperature=temperature, meta_agent=meta_agent)
        print(f"  Prompt {i+1}: extracted {len(concepts)} concepts")
        all_concepts.extend(concepts)
    
    # Deduplicate and normalize
    normalized = set(normalize_concept(c) for c in all_concepts if c.strip())
    final_concepts = list(normalized)
    
    print(f"Final result: {len(final_concepts)} unique concepts from {len(all_concepts)} total extractions")
    return final_concepts

def extract_concepts_with_metadata(text: str, n_prompts: int = CONCEPT_EXTRACTION_N_PROMPTS, 
                                  model: str = CONCEPT_EXTRACTION_MODEL, 
                                  temperature: float = CONCEPT_EXTRACTION_TEMPERATURE) -> Tuple[List[str], Dict[str, ConceptMetadata]]:
    """
    Extract concepts with full metadata tracking.
    
    Returns:
        Tuple of (final_concept_list, concept_metadata_dict)
    """
    all_concepts = []
    all_concept_metadata = {}
    
    # Initialize meta-prompting agent
    meta_agent = MetaPromptingAgent(model=model, temperature=temperature)
    
    print(f"Extracting concepts with metadata from full text ({len(text)} chars) with {n_prompts} prompt(s)...")
    
    # Show chunking info if text is long
    if len(text) > MAX_TEXT_LENGTH:
        estimated_chunks = len(chunk_text(text))
        print(f"Text is long, will use chunking strategy ({estimated_chunks} estimated chunks)")
    
    # Extract concepts multiple times
    for i in range(n_prompts):
        concepts, concept_metadata = llm_extract_concepts_with_context(
            text, model=model, temperature=temperature, meta_agent=meta_agent
        )
        print(f"  Prompt {i+1}: extracted {len(concepts)} concepts")
        
        # Merge concepts and metadata
        for concept in concepts:
            if concept not in all_concept_metadata:
                all_concept_metadata[concept] = ConceptMetadata(
                    concept=concept,
                    source_contexts=[],
                    chunk_indices=[]
                )
            
            # Merge metadata from this extraction
            if concept in concept_metadata:
                new_meta = concept_metadata[concept]
                all_concept_metadata[concept].source_contexts.extend(new_meta.source_contexts)
                all_concept_metadata[concept].chunk_indices.extend(new_meta.chunk_indices)
        
        all_concepts.extend(concepts)
    
    # Deduplicate and normalize
    normalized_concepts = set(normalize_concept(c) for c in all_concepts if c.strip())
    final_concepts = list(normalized_concepts)
    
    # Clean up metadata for final concepts
    final_metadata = {}
    for concept in final_concepts:
        # Find the best matching metadata
        for orig_concept, metadata in all_concept_metadata.items():
            if normalize_concept(orig_concept) == concept:
                if concept not in final_metadata:
                    final_metadata[concept] = ConceptMetadata(
                        concept=concept,
                        source_contexts=[],
                        chunk_indices=[]
                    )
                final_metadata[concept].source_contexts.extend(metadata.source_contexts)
                final_metadata[concept].chunk_indices.extend(metadata.chunk_indices)
    
    # Remove duplicate contexts
    for concept, metadata in final_metadata.items():
        metadata.source_contexts = list(set(metadata.source_contexts))
        metadata.chunk_indices = list(set(metadata.chunk_indices))
    
    print(f"Final result: {len(final_concepts)} unique concepts with metadata from {len(all_concepts)} total extractions")
    return final_concepts, final_metadata

if __name__ == "__main__":
    test_text = "Social isolation leads to poor sleep and depression."
    concepts = extract_concepts(test_text)
    print("Extracted concepts:", concepts) 
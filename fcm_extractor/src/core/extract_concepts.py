import os
import json
import time
from typing import List, Dict, Tuple
from dotenv import load_dotenv
import re
import sys

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models import MetaPromptingAgent, get_global_prompt_agent, TaskType
from config.constants import (
    CONCEPT_EXTRACTION_MODEL, CONCEPT_EXTRACTION_TEMPERATURE,
    CONCEPT_EXTRACTION_N_PROMPTS
)
from src.models.llm_client import llm_client
from src.models.cluster_metadata import ConceptMetadata

MAX_TEXT_LENGTH = 8000

def chunk_text(text: str, max_length: int = MAX_TEXT_LENGTH, overlap: int = 500) -> List[str]:
    if len(text) <= max_length:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + max_length
        
        if end < len(text):
            sentence_ends = ['.', '!', '?', '\n\n']
            best_break = end
            
            for i in range(end - 1000, end):
                if i > 0 and text[i] in sentence_ends:
                    best_break = i + 1
            
            end = best_break
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        start = end - overlap
        if start >= len(text):
            break
    
    return chunks

def llm_extract_concepts(text: str, model: str = CONCEPT_EXTRACTION_MODEL, 
                         temperature: float = CONCEPT_EXTRACTION_TEMPERATURE, meta_agent: MetaPromptingAgent = None) -> List[str]:
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
        
        unique_concepts = list(set(all_concepts))
        print(f"  Combined {len(all_concepts)} concepts into {len(unique_concepts)} unique concepts")
        return unique_concepts
    
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
        
        if not content.strip():
            print(f"    Warning: LLM returned empty response for text of length {len(text)}")
            return []
        
        concepts = parse_concepts_from_response(content)
        
        print(f"    Extracted {len(concepts)} concepts in {execution_time:.2f}s")
        return concepts
        
    except Exception as e:
        print(f"    Error in LLM concept extraction: {e}")
        return []

def parse_concepts_from_response(response: str) -> List[str]:
    if not response.strip():
        return []
    
    concepts = []
    lines = response.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        if ',' in line:
            parts = [part.strip() for part in line.split(',')]
            concepts.extend(parts)
        else:
            concepts.append(line)
    
    normalized_concepts = []
    for concept in concepts:
        normalized = normalize_concept(concept)
        if normalized and len(normalized) > 0:
            normalized_concepts.append(normalized)
    
    return list(set(normalized_concepts))

def llm_extract_concepts_with_context(text: str, chunk_index: int = 0, 
                                     model: str = CONCEPT_EXTRACTION_MODEL, 
                                     temperature: float = CONCEPT_EXTRACTION_TEMPERATURE, 
                                     meta_agent: MetaPromptingAgent = None) -> Tuple[List[str], Dict[str, ConceptMetadata]]:
    if len(text) > MAX_TEXT_LENGTH:
        print(f"  Text too long ({len(text)} chars), using chunking strategy...")
        chunks = chunk_text(text)
        print(f"  Split into {len(chunks)} chunks")
        
        all_concepts = []
        all_metadata = {}
        
        for i, chunk in enumerate(chunks):
            print(f"    Processing chunk {i+1}/{len(chunks)} ({len(chunk)} chars)...")
            chunk_concepts, chunk_metadata = llm_extract_concepts_with_context(
                chunk, i, model, temperature, meta_agent
            )
            all_concepts.extend(chunk_concepts)
            all_metadata.update(chunk_metadata)
            print(f"    Chunk {i+1} extracted {len(chunk_concepts)} concepts")
        
        unique_concepts = list(set(all_concepts))
        print(f"  Combined {len(all_concepts)} concepts into {len(unique_concepts)} unique concepts")
        
        combined_metadata = {}
        for concept in unique_concepts:
            if concept in all_metadata:
                combined_metadata[concept] = all_metadata[concept]
        
        return unique_concepts, combined_metadata
    
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
        
        if not content.strip():
            print(f"    Warning: LLM returned empty response for text of length {len(text)}")
            return [], {}
        
        concepts = parse_concepts_from_response(content)
        
        metadata = {}
        for concept in concepts:
            metadata[concept] = ConceptMetadata(
                concept=concept,
                chunk_index=chunk_index,
                text_length=len(text),
                extraction_time=execution_time,
                generation_method=generation_metadata.get('method', 'unknown'),
                prompt_length=len(adaptive_prompt),
                context_analysis=generation_metadata.get('context_analysis', {})
            )
        
        print(f"    Extracted {len(concepts)} concepts with metadata in {execution_time:.2f}s")
        return concepts, metadata
        
    except Exception as e:
        print(f"    Error in LLM concept extraction: {e}")
        return [], {}

def normalize_concept(concept: str) -> str:
    return concept.lower().strip()

def extract_concepts(text: str, n_prompts: int = CONCEPT_EXTRACTION_N_PROMPTS, model: str = CONCEPT_EXTRACTION_MODEL, 
                    temperature: float = CONCEPT_EXTRACTION_TEMPERATURE) -> List[str]:
    if not text or not text.strip():
        print("  Warning: Empty text provided for concept extraction")
        return []
    
    print(f"  📝 Extracting concepts from text ({len(text)} characters)...")
    
    all_concepts = []
    for i in range(n_prompts):
        print(f"    Prompt {i+1}/{n_prompts}...")
        concepts = llm_extract_concepts(text, model, temperature)
        all_concepts.extend(concepts)
    
    unique_concepts = list(set(all_concepts))
    print(f"  Extracted {len(unique_concepts)} unique concepts from {len(all_concepts)} total extractions")
    
    return unique_concepts

def extract_concepts_with_metadata(text: str, n_prompts: int = CONCEPT_EXTRACTION_N_PROMPTS, 
                                  model: str = CONCEPT_EXTRACTION_MODEL, 
                                  temperature: float = CONCEPT_EXTRACTION_TEMPERATURE) -> Tuple[List[str], Dict[str, ConceptMetadata]]:
    if not text or not text.strip():
        print("  Warning: Empty text provided for concept extraction")
        return [], {}
    
    print(f"  📝 Extracting concepts with metadata from text ({len(text)} characters)...")
    
    all_concepts = []
    all_metadata = {}
    
    for i in range(n_prompts):
        print(f"    Prompt {i+1}/{n_prompts}...")
        concepts, metadata = llm_extract_concepts_with_context(text, i, model, temperature)
        all_concepts.extend(concepts)
        all_metadata.update(metadata)
    
    unique_concepts = list(set(all_concepts))
    print(f"  Extracted {len(unique_concepts)} unique concepts from {len(all_concepts)} total extractions")
    
    combined_metadata = {}
    for concept in unique_concepts:
        if concept in all_metadata:
            combined_metadata[concept] = all_metadata[concept]
    
    return unique_concepts, combined_metadata 
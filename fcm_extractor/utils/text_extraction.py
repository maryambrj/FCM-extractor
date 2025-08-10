"""
Smart text extraction utilities for focused context around concept pairs.
Reduces context size while maintaining relevant information for edge inference.
"""

import re
from typing import List, Tuple, Dict, Optional
from config.constants import MAX_EDGE_INFERENCE_TEXT_LENGTH

class PairFocusedExtractor:
    """Extract text snippets focused on concept pairs to minimize context."""
    
    def __init__(self, max_length: int = MAX_EDGE_INFERENCE_TEXT_LENGTH):
        self.max_length = max_length
        
    def extract_pair_focused_text(
        self, 
        text: str, 
        concept_pairs: List[Tuple[str, str]], 
        context_sentences: int = 2
    ) -> str:
        """
        Extract focused text snippets around concept pairs.
        
        Args:
            text: Full text to extract from
            concept_pairs: List of (concept1, concept2) pairs
            context_sentences: Number of sentences before/after each match
            
        Returns:
            Focused text containing relevant snippets
        """
        # Early return if text is already short enough
        if len(text) <= self.max_length:
            return text
        
        # Split into sentences
        sentences = self._split_into_sentences(text)
        if not sentences:
            return text[:self.max_length]
        
        # Find relevant sentences for each concept pair
        relevant_indices = set()
        
        for concept1, concept2 in concept_pairs:
            pair_indices = self._find_relevant_sentences(
                sentences, concept1, concept2, context_sentences
            )
            relevant_indices.update(pair_indices)
        
        # If no specific matches, return truncated text
        if not relevant_indices:
            return self._truncate_text_intelligently(text)
        
        # Sort indices and extract sentences
        sorted_indices = sorted(relevant_indices)
        
        # Group consecutive indices to avoid repetition
        grouped_ranges = self._group_consecutive_indices(sorted_indices)
        
        # Extract text from grouped ranges
        extracted_parts = []
        for start_idx, end_idx in grouped_ranges:
            part = " ".join(sentences[start_idx:end_idx + 1])
            extracted_parts.append(part.strip())
        
        # Combine parts with separator
        focused_text = "\n\n".join(extracted_parts)
        
        # Final length check and truncation if needed
        if len(focused_text) > self.max_length:
            focused_text = self._truncate_text_intelligently(focused_text)
        
        return focused_text
    
    def extract_single_pair_text(
        self, 
        text: str, 
        concept1: str, 
        concept2: str,
        context_sentences: int = 3
    ) -> str:
        """
        Extract focused text for a single concept pair.
        More generous context for single-pair queries.
        """
        return self.extract_pair_focused_text(
            text, [(concept1, concept2)], context_sentences
        )
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using simple rules."""
        # Simple sentence splitting on periods, exclamation marks, question marks
        # Avoid splitting on abbreviations and decimals
        text = re.sub(r'\n+', ' ', text)  # Replace newlines with spaces
        
        # Split on sentence endings
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        
        # Filter out very short sentences
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        return sentences
    
    def _find_relevant_sentences(
        self, 
        sentences: List[str], 
        concept1: str, 
        concept2: str,
        context_sentences: int
    ) -> List[int]:
        """Find sentence indices relevant to a concept pair."""
        relevant_indices = []
        
        # Create flexible regex patterns for concepts
        c1_pattern = self._create_concept_pattern(concept1)
        c2_pattern = self._create_concept_pattern(concept2)
        
        for i, sentence in enumerate(sentences):
            sentence_lower = sentence.lower()
            
            # Check if sentence contains both concepts
            if (re.search(c1_pattern, sentence_lower) and 
                re.search(c2_pattern, sentence_lower)):
                # High priority: sentence contains both concepts
                start_idx = max(0, i - context_sentences)
                end_idx = min(len(sentences) - 1, i + context_sentences)
                relevant_indices.extend(range(start_idx, end_idx + 1))
            
            # Check if sentence contains either concept
            elif (re.search(c1_pattern, sentence_lower) or 
                  re.search(c2_pattern, sentence_lower)):
                # Medium priority: sentence contains one concept
                start_idx = max(0, i - context_sentences // 2)
                end_idx = min(len(sentences) - 1, i + context_sentences // 2)
                relevant_indices.extend(range(start_idx, end_idx + 1))
        
        return list(set(relevant_indices))  # Remove duplicates
    
    def _create_concept_pattern(self, concept: str) -> str:
        """Create a flexible regex pattern for concept matching."""
        # Handle multi-word concepts
        concept_words = concept.lower().split()
        
        if len(concept_words) == 1:
            # Single word: match whole word with plural/variations
            word = re.escape(concept_words[0])
            return rf'\b{word}[s]?\b'
        else:
            # Multi-word: allow some flexibility in word order and separators
            escaped_words = [re.escape(word) for word in concept_words]
            
            # Try exact phrase first, then flexible matching
            exact_phrase = r'\b' + r'\s+'.join(escaped_words) + r'\b'
            flexible_pattern = r'\b(?:' + '|'.join(escaped_words) + r')\b.*\b(?:' + '|'.join(escaped_words) + r')\b'
            
            return f'(?:{exact_phrase}|{flexible_pattern})'
    
    def _group_consecutive_indices(self, indices: List[int]) -> List[Tuple[int, int]]:
        """Group consecutive indices into ranges."""
        if not indices:
            return []
        
        ranges = []
        start = indices[0]
        end = indices[0]
        
        for i in indices[1:]:
            if i == end + 1:
                # Consecutive
                end = i
            else:
                # Gap found, save current range and start new one
                ranges.append((start, end))
                start = i
                end = i
        
        # Add the last range
        ranges.append((start, end))
        
        return ranges
    
    def _truncate_text_intelligently(self, text: str) -> str:
        """Truncate text intelligently, preserving sentence boundaries."""
        if len(text) <= self.max_length:
            return text
        
        # Try to truncate at sentence boundary
        truncated = text[:self.max_length]
        
        # Find last complete sentence
        last_period = truncated.rfind('.')
        last_exclamation = truncated.rfind('!')
        last_question = truncated.rfind('?')
        
        last_sentence_end = max(last_period, last_exclamation, last_question)
        
        if last_sentence_end > self.max_length * 0.8:  # At least 80% of max length
            return truncated[:last_sentence_end + 1]
        else:
            # If no good sentence boundary, truncate at word boundary
            last_space = truncated.rfind(' ')
            if last_space > self.max_length * 0.9:  # At least 90% of max length
                return truncated[:last_space] + "..."
            else:
                return truncated + "..."

def enforce_text_length_limit(text: str, limit: int = MAX_EDGE_INFERENCE_TEXT_LENGTH) -> str:
    """
    Enforce text length limit with intelligent truncation.
    Simple utility function for general use.
    """
    if len(text) <= limit:
        return text
    
    extractor = PairFocusedExtractor(limit)
    return extractor._truncate_text_intelligently(text)

def extract_focused_context(
    text: str, 
    concept_pairs: List[Tuple[str, str]],
    max_length: int = MAX_EDGE_INFERENCE_TEXT_LENGTH
) -> str:
    """
    High-level function to extract focused context for concept pairs.
    
    Args:
        text: Full text to extract from
        concept_pairs: List of (concept1, concept2) pairs to focus on
        max_length: Maximum allowed text length
        
    Returns:
        Focused text optimized for edge inference
    """
    extractor = PairFocusedExtractor(max_length)
    return extractor.extract_pair_focused_text(text, concept_pairs)

# =============================================================================
# TESTING AND EXAMPLES
# =============================================================================

def test_pair_focused_extraction():
    """Test the pair-focused text extraction."""
    
    # Sample text
    sample_text = """
    Social isolation is a growing problem in modern society. Research shows that social isolation 
    can lead to significant mental health issues including depression and anxiety. Studies have found
    that isolated individuals are more likely to develop these conditions.
    
    Regular physical exercise has been proven to be an effective treatment for depression. Exercise
    releases endorphins which improve mood and reduce stress levels. Many therapists now recommend
    exercise as part of treatment plans for patients with anxiety disorders.
    
    Sleep quality is another important factor in mental health. Poor sleep can worsen depression
    and anxiety symptoms. Establishing good sleep hygiene practices can help improve overall
    mental well-being.
    
    Social support networks play a crucial role in preventing isolation. Community programs that
    bring people together have shown positive results in reducing loneliness and improving
    mental health outcomes.
    """
    
    # Test concept pairs
    concept_pairs = [
        ("social isolation", "depression"),
        ("exercise", "anxiety"),
        ("sleep quality", "mental health")
    ]
    
    print("=== PAIR-FOCUSED TEXT EXTRACTION TEST ===")
    print(f"Original text length: {len(sample_text)} characters")
    
    # Extract focused text
    extractor = PairFocusedExtractor(max_length=1000)  # Small limit for testing
    focused_text = extractor.extract_pair_focused_text(sample_text, concept_pairs)
    
    print(f"Focused text length: {len(focused_text)} characters")
    print("\\nFocused text:")
    print("-" * 50)
    print(focused_text)
    print("-" * 50)
    
    # Test single pair
    single_pair_text = extractor.extract_single_pair_text(sample_text, "exercise", "depression")
    print(f"\\nSingle pair text length: {len(single_pair_text)} characters")
    print("Single pair text:")
    print("-" * 30)
    print(single_pair_text)
    
    return focused_text

if __name__ == "__main__":
    test_pair_focused_extraction()
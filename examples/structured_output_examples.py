#!/usr/bin/env python3
"""
Examples of using Pydantic structured outputs with the FCM Extractor.

This file demonstrates how to use the new structured output functionality
to get consistent, validated responses across different OpenAI models.
"""

import os
import sys
from typing import List, Tuple

# Add the fcm_extractor to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'fcm_extractor'))

from src.models.llm_client import llm_client
from src.models.pydantic_schemas import (
    ConceptExtractionResponse, 
    BatchEdgeInferenceResponse, 
    EdgeRelationship,
    LegacyFormatParser
)


def example_concept_extraction():
    """Example of structured concept extraction."""
    print("📝 Example: Structured Concept Extraction")
    print("=" * 50)
    
    sample_text = """
    The increased workload at the office has been causing me significant stress. 
    This stress is directly impacting my sleep quality - I find myself lying awake 
    at night worrying about deadlines. Poor sleep then affects my concentration 
    during important meetings the next day. When I can't focus properly, I tend 
    to make more mistakes, which creates even more stress. It's become a vicious 
    cycle that's affecting my overall job performance and satisfaction.
    """
    
    # Test with different models
    models_to_test = ["gpt-4o", "gpt-4", "gpt-4-turbo"]
    
    for model in models_to_test:
        print(f"\n🔍 Testing with {model}:")
        
        messages = [
            {"role": "system", "content": "Extract key concepts from the text."},
            {"role": "user", "content": f"Text: {sample_text.strip()}"}
        ]
        
        try:
            # Using structured completion
            response, confidence = llm_client.structured_completion(
                model, messages, ConceptExtractionResponse
            )
            
            print(f"  ✅ Structured output successful!")
            print(f"  📊 Extracted {len(response.concepts)} concepts:")
            for concept in response.concepts[:5]:  # Show first 5
                print(f"     - {concept}")
            if len(response.concepts) > 5:
                print(f"     ... and {len(response.concepts) - 5} more")
            print(f"  🎯 Confidence: {confidence}")
            
        except Exception as e:
            print(f"  ❌ Failed: {e}")


def example_edge_inference():
    """Example of structured edge inference."""
    print("\n\n🔗 Example: Structured Edge Inference")
    print("=" * 50)
    
    sample_text = """
    When I exercise regularly, I notice my energy levels increase significantly. 
    Higher energy levels help me stay more focused during work tasks. Better focus 
    leads to improved productivity and quality of work. Good work performance 
    reduces my stress levels, which makes me more motivated to continue exercising.
    """
    
    concept_pairs = [
        ("exercise", "energy levels"),
        ("energy levels", "focus"),
        ("focus", "productivity"),
        ("productivity", "stress"),
        ("stress", "motivation")
    ]
    
    model = "gpt-4o"  # Use a model with good structured output support
    print(f"🔍 Testing edge inference with {model}:")
    
    # Prepare the prompt
    pairs_text = "\n".join([f"Pair {i+1}: '{p[0]}' and '{p[1]}'" for i, p in enumerate(concept_pairs)])
    
    messages = [
        {
            "role": "system", 
            "content": "Analyze causal relationships between concept pairs based on the provided text."
        },
        {
            "role": "user", 
            "content": f"Text: {sample_text.strip()}\n\nConcept pairs to analyze:\n{pairs_text}"
        }
    ]
    
    try:
        # Using structured completion with pairs parameter
        response, confidence = llm_client.structured_completion(
            model, messages, BatchEdgeInferenceResponse, pairs=concept_pairs
        )
        
        print(f"  ✅ Structured output successful!")
        print(f"  📊 Analyzed {len(response.pair_analyses)} pairs:")
        
        relationships = response.to_edge_relationships()
        print(f"  🔗 Found {len(relationships)} causal relationships:")
        
        for rel in relationships:
            arrow = "→" if rel.relationship_type == "positive" else "⊣"
            print(f"     {rel.source_concept} {arrow} {rel.target_concept}")
            print(f"       Type: {rel.relationship_type}, Confidence: {rel.confidence:.2f}")
        
        print(f"  🎯 Overall confidence: {confidence}")
        
    except Exception as e:
        print(f"  ❌ Failed: {e}")


def example_legacy_parsing():
    """Example of legacy format parsing for fallback compatibility."""
    print("\n\n🔄 Example: Legacy Format Parsing")
    print("=" * 50)
    
    # Simulate legacy text responses that might come from older models
    legacy_concept_response = """
    1. work stress
    2. sleep quality  
    3. concentration problems
    4. meeting performance
    5. job satisfaction
    """
    
    legacy_edge_response = """
    Pair 1: stress -> sleep quality (negative, confidence: 0.85)
    Pair 2: sleep quality -> concentration (positive, confidence: 0.80)
    Pair 3: no relationship
    Pair 4: concentration -> mistakes (negative, confidence: 0.75)
    """
    
    concept_pairs = [
        ("stress", "sleep quality"),
        ("sleep quality", "concentration"), 
        ("concentration", "performance"),
        ("concentration", "mistakes")
    ]
    
    print("📝 Parsing legacy concept extraction response:")
    concept_response = LegacyFormatParser.parse_concept_extraction(legacy_concept_response)
    print(f"  ✅ Parsed {len(concept_response.concepts)} concepts:")
    for concept in concept_response.concepts:
        print(f"     - {concept}")
    
    print("\n🔗 Parsing legacy edge inference response:")
    edge_response = LegacyFormatParser.parse_edge_inference(legacy_edge_response, concept_pairs)
    print(f"  ✅ Parsed {len(edge_response.pair_analyses)} pair analyses:")
    
    for analysis in edge_response.pair_analyses:
        if analysis.has_relationship:
            print(f"     {analysis.source_concept} → {analysis.target_concept}")
            print(f"       {analysis.relationship_type} relationship (confidence: {analysis.confidence})")
        else:
            print(f"     {analysis.source_concept} ⊗ {analysis.target_concept} (no relationship)")


def example_error_handling():
    """Example of error handling and fallback mechanisms."""
    print("\n\n🛡️  Example: Error Handling & Fallback")
    print("=" * 50)
    
    # Test with a model that doesn't support structured output
    model = "gpt-4-turbo"  # This model falls back to text parsing
    
    messages = [
        {"role": "system", "content": "Extract concepts from the text."},
        {"role": "user", "content": "Text: Stress affects sleep, which impacts concentration."}
    ]
    
    print(f"🔍 Testing fallback with {model} (doesn't support structured output):")
    
    try:
        response, confidence = llm_client.structured_completion(
            model, messages, ConceptExtractionResponse
        )
        
        print(f"  ✅ Fallback successful!")
        print(f"  📊 Concepts via text parsing: {response.concepts}")
        print(f"  🎯 Confidence: {confidence}")
        print(f"  💡 Note: This used text parsing fallback, not native structured output")
        
    except Exception as e:
        print(f"  ❌ Error: {e}")


def main():
    """Run all structured output examples."""
    print("🚀 FCM Extractor: Pydantic Structured Output Examples")
    print("=" * 60)
    print("This script demonstrates the new structured output functionality")
    print("that ensures consistent responses across different OpenAI models.")
    print("=" * 60)
    
    try:
        example_concept_extraction()
        example_edge_inference() 
        example_legacy_parsing()
        example_error_handling()
        
        print("\n" + "=" * 60)
        print("✅ All examples completed successfully!")
        print("💡 The structured output system provides:")
        print("   - Consistent data formats across models")
        print("   - Automatic validation with Pydantic")
        print("   - Graceful fallback for unsupported models")
        print("   - Better error handling and debugging")
        
    except Exception as e:
        print(f"\n❌ Example failed with error: {e}")
        print("💡 Make sure you have:")
        print("   - Set OPENAI_API_KEY environment variable")
        print("   - Installed all required dependencies")
        print("   - Run from the correct directory")


if __name__ == "__main__":
    main()
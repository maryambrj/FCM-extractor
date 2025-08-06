import os
import json
from typing import List, Dict, Tuple, Any
from dotenv import load_dotenv
import re
import sys

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))

# Add parent directories to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.constants import (
    META_PROMPTING_MODEL, META_PROMPTING_TEMPERATURE, META_PROMPTING_VERBOSE,
    META_PROMPTING_ENABLED, DEFAULT_CONCEPT_EXTRACTION_PROMPT, DEFAULT_EDGE_INFERENCE_PROMPT
)
from src.models.llm_client import llm_client


class MetaPromptingAgent:
    """Agent that dynamically decides what prompts to use based on context."""
    
    def __init__(self, model: str = META_PROMPTING_MODEL, temperature: float = META_PROMPTING_TEMPERATURE):
        self.model = model
        self.temperature = temperature
        
        # Track previous attempts for better decision making
        self.previous_attempts = {}
        
        # Base prompt templates for different tasks
        self.base_prompts = {
            "concept_extraction": [
                "Extract all key concepts or entities from the following text:",
                "List the main noun phrases present in this passage:",
                "Identify important factors or variables mentioned in the text:",
                "What are the key variables or concepts in this passage?",
                "Extract the main topics and entities from this text:"
            ],
            "edge_inference": [
                "Does the text suggest that [A] causes [B], [B] causes [A], or neither?",
                "Based on the passage, what is the direction of influence between [A] and [B]?",
                "Is there a causal relationship between [A] and [B] in the text? If so, which direction?",
                "Analyze the causal relationship between [A] and [B] in this context:",
                "What is the relationship between [A] and [B]? Does one influence the other?"
            ]
        }
        
        # Meta-prompt for deciding which prompt to use
        self.meta_prompt = """You are a meta-prompting agent that decides which prompt template to use for a given task.

Available prompt templates for {task_type}:
{prompts}

Context:
- Text: {text}
- Previous attempts: {previous_attempts}
- Current concepts: {concepts}
- Target concepts: {target_concepts}

Based on the context, which prompt template (number 1-{num_prompts}) would be most effective for this specific case? 
Consider the complexity of the text, the nature of the concepts, and any patterns in previous attempts.

Respond with only the number (1-{num_prompts}) and a brief reason (max 50 words)."""

    def decide_prompt(self, task_type: str, text: str, concepts: List[str] = None, 
                     target_concepts: List[str] = None, previous_attempts: List[Dict] = None) -> Tuple[str, str]:
        """Decide which prompt template to use for the given task."""
        if task_type not in self.base_prompts:
            raise ValueError(f"Unknown task type: {task_type}")
        
        prompts = self.base_prompts[task_type]
        num_prompts = len(prompts)
        
        # Get stored previous attempts for this task type
        stored_attempts = self.previous_attempts.get(task_type, [])
        if previous_attempts:
            stored_attempts.extend(previous_attempts)
        
        # Format previous attempts for context
        attempts_str = "None" if not stored_attempts else str(stored_attempts[-3:])  # Last 3 attempts
        
        # Build meta-prompt
        meta_prompt = self.meta_prompt.format(
            task_type=task_type,
            prompts="\n".join([f"{i+1}. {prompt}" for i, prompt in enumerate(prompts)]),
            text=text[:500],  # Limit text length
            concepts=str(concepts) if concepts else "None",
            target_concepts=str(target_concepts) if target_concepts else "None",
            previous_attempts=attempts_str,
            num_prompts=num_prompts
        )
        
        try:
            messages = [
                {"role": "system", "content": "You are a meta-prompting agent. Respond with only a number and brief reason."},
                {"role": "user", "content": meta_prompt}
            ]
            
            content = llm_client.chat_completion(self.model, messages, self.temperature, max_tokens=100)
            
            # Parse response to get prompt number
            lines = content.split('\n')
            first_line = lines[0].strip()
            
            # Extract number from response
            number_match = re.search(r'(\d+)', first_line)
            if number_match:
                prompt_idx = int(number_match.group(1)) - 1  # Convert to 0-based index
                if 0 <= prompt_idx < num_prompts:
                    selected_prompt = prompts[prompt_idx]
                    reason = content if len(content) > 10 else "No specific reason given"
                    return selected_prompt, reason
            
            # Fallback to first prompt if parsing fails
            return prompts[0], f"Fallback used. Original response: {content}"
            
        except Exception as e:
            print(f"Warning: Meta-prompting failed: {e}")
            return prompts[0], "Fallback due to error"
        
        finally:
            # Store this attempt for future reference
            if task_type not in self.previous_attempts:
                self.previous_attempts[task_type] = []
            self.previous_attempts[task_type].append({
                "text": text[:100],  # Store truncated text
                "concepts": concepts,
                "target_concepts": target_concepts,
                "timestamp": "now"  # Could use actual timestamp if needed
            })

    def get_adaptive_prompt(self, task_type: str, text: str, **kwargs) -> str:
        """Get an adaptive prompt based on the current context."""
        if not META_PROMPTING_ENABLED:
            # Use default prompts when meta-prompting is disabled
            if task_type == "concept_extraction":
                return DEFAULT_CONCEPT_EXTRACTION_PROMPT
            elif task_type == "edge_inference":
                return DEFAULT_EDGE_INFERENCE_PROMPT
            else:
                # Fallback to first prompt in the list
                return self.base_prompts[task_type][0]
        
        # Meta-prompting is enabled, use dynamic selection
        selected_prompt, reason = self.decide_prompt(task_type, text, **kwargs)
        if META_PROMPTING_VERBOSE:
            print(f"Meta-prompting selected: {reason}")
        return selected_prompt

if __name__ == "__main__":
    # Test the meta-prompting agent
    agent = MetaPromptingAgent(model=META_PROMPTING_MODEL, temperature=META_PROMPTING_TEMPERATURE)
    text = "Social isolation leads to poor sleep and depression."
    prompt = agent.get_adaptive_prompt("edge_inference", text, 
                                     concepts=["social isolation", "depression"],
                                     target_concepts=["social isolation", "depression"])
    print(f"Selected prompt: {prompt}") 
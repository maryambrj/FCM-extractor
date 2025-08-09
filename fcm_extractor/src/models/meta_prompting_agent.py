import os
import json
from typing import List, Dict, Tuple, Any
from dotenv import load_dotenv
import re
import sys

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.constants import (
    META_PROMPTING_MODEL, META_PROMPTING_TEMPERATURE, META_PROMPTING_VERBOSE,
    META_PROMPTING_ENABLED, DEFAULT_CONCEPT_EXTRACTION_PROMPT, DEFAULT_EDGE_INFERENCE_PROMPT
)
from utils.llm_utils import supports_temperature
from src.models.llm_client import llm_client


class MetaPromptingAgent:
    def __init__(self, model: str = META_PROMPTING_MODEL, temperature: float = META_PROMPTING_TEMPERATURE):
        self.model = model
        self.temperature = temperature
        
        self.previous_attempts = {}
        
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
        if task_type not in self.base_prompts:
            raise ValueError(f"Unknown task type: {task_type}")
        
        prompts = self.base_prompts[task_type]
        num_prompts = len(prompts)
        
        stored_attempts = self.previous_attempts.get(task_type, [])
        if previous_attempts:
            stored_attempts.extend(previous_attempts)
        
        attempts_str = "None" if not stored_attempts else str(stored_attempts[-3:])
        
        meta_prompt = self.meta_prompt.format(
            task_type=task_type,
            prompts="\n".join([f"{i+1}. {prompt}" for i, prompt in enumerate(prompts)]),
            text=text[:500],
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
            
            # Use temperature only if supported by the model
            temp = self.temperature if supports_temperature(self.model) else 0.0
            content = llm_client.chat_completion(self.model, messages, temp, max_tokens=100)
            
            if META_PROMPTING_VERBOSE:
                print(f"Meta-prompting decision for {task_type}: {content}")
            
            return content, meta_prompt
            
        except Exception as e:
            print(f"Error in meta-prompting: {e}")
            return f"1. Default prompt (error: {e})", meta_prompt

    def get_adaptive_prompt(self, task_type: str, text: str, **kwargs) -> str:
        decision, meta_prompt = self.decide_prompt(task_type, text, **kwargs)
        
        try:
            prompt_num = int(decision.split('.')[0]) - 1
            if 0 <= prompt_num < len(self.base_prompts[task_type]):
                return self.base_prompts[task_type][prompt_num]
        except (ValueError, IndexError):
            pass
        
        return self.base_prompts[task_type][0] 
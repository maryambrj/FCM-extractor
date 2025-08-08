import os
import json
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum
import re
import sys
from collections import defaultdict
import hashlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.constants import (
    META_PROMPTING_MODEL, META_PROMPTING_TEMPERATURE, META_PROMPTING_VERBOSE,
    META_PROMPTING_ENABLED, LOG_DIRECTORY, DYNAMIC_PROMPTING_ENABLED,
    DYNAMIC_PROMPTING_USE_CACHE, DYNAMIC_PROMPTING_USE_REFLECTION,
    DYNAMIC_PROMPTING_TRACK_PERFORMANCE
)
from src.models.llm_client import llm_client


class TaskType(Enum):
    CONCEPT_EXTRACTION = "concept_extraction"
    EDGE_INFERENCE = "edge_inference"
    CLUSTERING = "clustering"
    POST_CLUSTERING = "post_clustering"
    EVALUATION = "evaluation"


class TextComplexity(Enum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    HIGHLY_COMPLEX = "highly_complex"


@dataclass
class TextAnalysis:
    length: int
    sentence_count: int
    avg_sentence_length: float
    complexity_score: float
    domain_indicators: List[str]
    technical_terms_count: int
    narrative_style: str
    emotional_content: str
    causal_language_count: int
    concept_density: float
    text_hash: str


@dataclass
class PromptPerformance:
    prompt_hash: str
    task_type: TaskType
    text_analysis: TextAnalysis
    performance_score: float
    execution_time: float
    output_quality: float
    timestamp: datetime
    success: bool
    error_message: Optional[str] = None


class DynamicPromptAgent:
    def __init__(self, model: str = META_PROMPTING_MODEL, temperature: float = META_PROMPTING_TEMPERATURE):
        self.model = model
        self.temperature = temperature
        
        self.performance_history: List[PromptPerformance] = []
        self.prompt_cache: Dict[str, str] = {}
        self.adaptation_rate = 0.1
        
        self.domain_vocabularies = self._load_domain_vocabularies()
        self.causal_indicators = self._load_causal_indicators()
        self.task_objectives = self._load_task_objectives()
        
        self._load_performance_history()
    
    def _load_domain_vocabularies(self) -> Dict[str, List[str]]:
        return {
            'psychological': [
                'stress', 'anxiety', 'depression', 'mood', 'emotion', 'behavior', 'cognitive', 
                'mental health', 'wellbeing', 'psychological', 'therapy', 'counseling',
                'resilience', 'coping', 'trauma', 'self-esteem', 'motivation'
            ],
            'medical': [
                'symptoms', 'diagnosis', 'treatment', 'patient', 'clinical', 'health',
                'disease', 'condition', 'medical', 'therapeutic', 'intervention',
                'outcome', 'prognosis', 'comorbidity', 'medication', 'dosage'
            ],
            'social': [
                'relationship', 'social', 'community', 'interaction', 'group', 'family',
                'peer', 'support', 'network', 'cultural', 'society', 'interpersonal',
                'communication', 'collaboration', 'conflict', 'trust'
            ],
            'organizational': [
                'management', 'leadership', 'team', 'organization', 'workplace', 'employee',
                'performance', 'productivity', 'culture', 'structure', 'process',
                'efficiency', 'strategy', 'goals', 'objectives', 'resources'
            ],
            'educational': [
                'learning', 'student', 'teacher', 'education', 'academic', 'curriculum',
                'assessment', 'pedagogy', 'instruction', 'knowledge', 'skill',
                'achievement', 'motivation', 'engagement', 'feedback'
            ]
        }
    
    def _load_causal_indicators(self) -> List[str]:
        return [
            'causes', 'leads to', 'results in', 'because of', 'due to', 'owing to',
            'triggers', 'influences', 'affects', 'impacts', 'contributes to',
            'brings about', 'gives rise to', 'stems from', 'derives from',
            'consequently', 'therefore', 'thus', 'hence', 'as a result',
            'enables', 'facilitates', 'promotes', 'inhibits', 'prevents',
            'mediates', 'moderates', 'predicts', 'determines'
        ]
    
    def _load_task_objectives(self) -> Dict[TaskType, Dict]:
        return {
            TaskType.CONCEPT_EXTRACTION: {
                'primary_goal': 'Extract semantically meaningful concepts that capture key variables and entities',
                'success_criteria': ['completeness', 'relevance', 'granularity', 'uniqueness'],
                'avoid': ['trivial terms', 'stop words', 'overly specific phrases', 'redundancy'],
                'output_format': 'comma-separated list of 1-3 word concepts'
            },
            TaskType.EDGE_INFERENCE: {
                'primary_goal': 'Identify causal relationships with accurate direction and strength',
                'success_criteria': ['causal accuracy', 'direction correctness', 'confidence calibration'],
                'avoid': ['correlation as causation', 'reverse causation errors', 'spurious relationships'],
                'output_format': 'structured relationship assessments with confidence scores'
            },
            TaskType.CLUSTERING: {
                'primary_goal': 'Group semantically related concepts into meaningful clusters',
                'success_criteria': ['semantic coherence', 'appropriate granularity', 'balanced sizes'],
                'avoid': ['trivial clusters', 'overly broad categories', 'conceptual mixing'],
                'output_format': 'cluster assignments with semantic labels'
            },
            TaskType.POST_CLUSTERING: {
                'primary_goal': 'Merge isolated concepts with semantically similar connected clusters',
                'success_criteria': ['semantic similarity', 'cluster coherence', 'graph connectivity'],
                'avoid': ['forced merging', 'semantic drift', 'over-clustering'],
                'output_format': 'merge recommendations with similarity scores'
            }
        }
    
    def analyze_text(self, text: str) -> TextAnalysis:
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        words = text.lower().split()
        
        length = len(text)
        sentence_count = len(sentences)
        avg_sentence_length = length / max(sentence_count, 1)
        
        domain_scores = {}
        for domain, vocab in self.domain_vocabularies.items():
            score = sum(1 for word in vocab if word.lower() in text.lower())
            domain_scores[domain] = score
        
        top_domains = sorted(domain_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        domain_indicators = [domain for domain, score in top_domains if score > 0]
        
        technical_terms = sum(1 for word in words if len(word) > 8 and word.isalpha())
        
        first_person = len(re.findall(r'\b(I|me|my|we|us|our)\b', text, re.IGNORECASE))
        structured_indicators = len(re.findall(r'\b(first|second|third|step|stage|phase)\b', text, re.IGNORECASE))
        
        if first_person > structured_indicators:
            narrative_style = "narrative"
        elif structured_indicators > first_person:
            narrative_style = "structured"
        else:
            narrative_style = "mixed"
        
        emotional_words = ['feel', 'emotion', 'happy', 'sad', 'angry', 'fear', 'joy', 'stress', 'anxiety']
        emotional_count = sum(1 for word in emotional_words if word in text.lower())
        emotional_content = "high" if emotional_count > 5 else "medium" if emotional_count > 2 else "low"
        
        causal_count = sum(1 for indicator in self.causal_indicators if indicator in text.lower())
        
        complexity_factors = [
            min(avg_sentence_length / 20, 1),
            min(technical_terms / len(words) * 10, 1),
            min(len(set(words)) / len(words) * 2, 1),
            min(causal_count / 10, 1)
        ]
        complexity_score = sum(complexity_factors) * 2.5
        
        potential_concepts = len([w for w in words if len(w) > 3 and w.isalpha()])
        concept_density = potential_concepts / len(words)
        
        text_hash = hashlib.md5(text.encode()).hexdigest()[:12]
        
        return TextAnalysis(
            length=length,
            sentence_count=sentence_count,
            avg_sentence_length=avg_sentence_length,
            complexity_score=complexity_score,
            domain_indicators=domain_indicators,
            technical_terms_count=technical_terms,
            narrative_style=narrative_style,
            emotional_content=emotional_content,
            causal_language_count=causal_count,
            concept_density=concept_density,
            text_hash=text_hash
        )
    
    def generate_dynamic_prompt(self, task_type: TaskType, text: str, **kwargs) -> Tuple[str, Dict]:
        if not META_PROMPTING_ENABLED or not DYNAMIC_PROMPTING_ENABLED:
            return self._get_default_prompt(task_type), {'method': 'default'}
        
        text_analysis = self.analyze_text(text)
        
        cache_key = f"{task_type.value}_{text_analysis.text_hash}_{str(kwargs)}"
        if DYNAMIC_PROMPTING_USE_CACHE and cache_key in self.prompt_cache:
            if META_PROMPTING_VERBOSE:
                cache_msg = f"💾 Using cached prompt for {task_type.value} (cache key: {cache_key[:20]}...)"
                print(cache_msg)
                self._log_to_file(cache_msg)
            return self.prompt_cache[cache_key], {'method': 'cached', 'text_analysis': text_analysis}
        
        performance_insights = self._get_performance_insights(task_type, text_analysis)
        
        generated_prompt = self._generate_contextual_prompt(task_type, text_analysis, performance_insights, **kwargs)
        
        if DYNAMIC_PROMPTING_USE_REFLECTION:
            if META_PROMPTING_VERBOSE:
                reflection_msg = f"🔍 Applying self-reflection for {task_type.value} prompt refinement..."
                print(reflection_msg)
                self._log_to_file(reflection_msg)
            refined_prompt = self._refine_prompt_with_reflection(generated_prompt, task_type, text_analysis)
            if META_PROMPTING_VERBOSE and refined_prompt != generated_prompt:
                refinement_msg = f"✨ Prompt refined through self-reflection (length: {len(generated_prompt)} → {len(refined_prompt)} chars)"
                print(refinement_msg)
                self._log_to_file(refinement_msg)
        else:
            refined_prompt = generated_prompt
        
        if DYNAMIC_PROMPTING_USE_CACHE:
            self.prompt_cache[cache_key] = refined_prompt
        
        generation_metadata = {
            'method': 'dynamic_generation',
            'text_analysis': text_analysis,
            'performance_insights': performance_insights,
            'cache_key': cache_key
        }
        
        if META_PROMPTING_VERBOSE:
            verbose_output = f"DYNAMIC PROMPT GENERATION REPORT for {task_type.value.upper()}\n"
            verbose_output += f"{'='*80}\n"
            verbose_output += f"TEXT ANALYSIS:\n"
            verbose_output += f"   • Length: {text_analysis.length} characters ({text_analysis.sentence_count} sentences)\n"
            verbose_output += f"   • Complexity Score: {text_analysis.complexity_score:.1f}/10\n"
            verbose_output += f"   • Domains: {', '.join(text_analysis.domain_indicators) if text_analysis.domain_indicators else 'general'}\n"
            verbose_output += f"   • Style: {text_analysis.narrative_style}\n"
            verbose_output += f"   • Emotional Content: {text_analysis.emotional_content}\n"
            verbose_output += f"   • Causal Language Count: {text_analysis.causal_language_count}\n"
            verbose_output += f"   • Technical Terms: {text_analysis.technical_terms_count}\n"
            verbose_output += f"   • Concept Density: {text_analysis.concept_density:.2f}\n"
            verbose_output += f"\nPERFORMANCE INSIGHTS:\n"
            verbose_output += f"   {self._format_performance_insights(performance_insights).replace(chr(10), chr(10) + '   ')}\n"
            verbose_output += f"\nGENERATION METHOD: {generation_metadata['method']}\n"
            verbose_output += f"📏 FINAL PROMPT LENGTH: {len(refined_prompt)} characters\n"
            verbose_output += f"\nGENERATED PROMPT:\n"
            verbose_output += f"{'-'*80}\n{refined_prompt}\n{'-'*80}"
            
            print(verbose_output)
            self._log_to_file(verbose_output)
        
        return refined_prompt, generation_metadata
    
    def _generate_contextual_prompt(self, task_type: TaskType, text_analysis: TextAnalysis, 
                                  performance_insights: Dict, **kwargs) -> str:
        
        meta_prompt = f"""Generate a precise prompt for {task_type.value} that is optimized for the following context:

TEXT CHARACTERISTICS:
- Length: {text_analysis.length} chars, {text_analysis.sentence_count} sentences
- Complexity: {text_analysis.complexity_score:.1f}/10
- Domains: {', '.join(text_analysis.domain_indicators) if text_analysis.domain_indicators else 'general'}
- Style: {text_analysis.narrative_style}
- Emotional content: {text_analysis.emotional_content}
- Technical terms: {text_analysis.technical_terms_count}

TASK REQUIREMENTS:
- Goal: {self.task_objectives[task_type]['primary_goal']}
- Success criteria: {', '.join(self.task_objectives[task_type]['success_criteria'])}
- Avoid: {', '.join(self.task_objectives[task_type]['avoid'])}
- Output format: {self.task_objectives[task_type]['output_format']}

PERFORMANCE DATA:
{self._format_performance_insights(performance_insights)}

CONSTRAINTS: {self._format_additional_context(kwargs)}

Create a direct, actionable prompt that:
1. Addresses the specific domains and complexity level
2. Provides clear, unambiguous instructions
3. Specifies the exact output format required
4. Uses appropriate terminology for the detected domains

Return ONLY the prompt text, no explanations or analysis."""

        try:
            messages = [
                {"role": "system", "content": "You are an expert prompt engineer. Your response must contain ONLY the requested prompt text with no additional commentary, explanations, or analysis. Do not include phrases like 'Here is the prompt:' or 'Certainly!' - just return the direct prompt."},
                {"role": "user", "content": meta_prompt}
            ]
            
            response, _ = llm_client.chat_completion(self.model, messages, self.temperature, max_tokens=1000)
            
            generated_prompt = response.strip()
            
            unwanted_prefixes = [
                "Here is the prompt:", "Here's the prompt:", "Certainly!", "Here is a", "Here's a",
                "The prompt is:", "Generated prompt:", "GENERATED PROMPT:", "**Refined Prompt:**",
                "**Generated Prompt:**", "Refined Prompt:", "Generated Prompt:"
            ]
            
            for prefix in unwanted_prefixes:
                if generated_prompt.startswith(prefix):
                    generated_prompt = generated_prompt[len(prefix):].strip()
                    break
            
            return generated_prompt
            
        except Exception as e:
            print(f"Warning: Dynamic prompt generation failed: {e}")
            return self._get_default_prompt(task_type)
    
    def _refine_prompt_with_reflection(self, initial_prompt: str, task_type: TaskType, 
                                     text_analysis: TextAnalysis) -> str:
        
        reflection_prompt = f"""Improve this {task_type.value} prompt for better clarity and effectiveness:

CURRENT PROMPT:
{initial_prompt}

CONTEXT:
- Domains: {', '.join(text_analysis.domain_indicators)}
- Complexity: {text_analysis.complexity_score:.1f}/10
- Style: {text_analysis.narrative_style}

Make the prompt more specific, clear, and effective while maintaining its core purpose. Focus on:
1. Domain-specific terminology where appropriate
2. Clear output format specification
3. Reduced ambiguity
4. Better structure for LLM processing

Return ONLY the improved prompt text, no analysis or explanation."""

        try:
            messages = [
                {"role": "system", "content": "You are a prompt refinement expert. Return ONLY the improved prompt text with no additional commentary or analysis. Do not include phrases like 'Here is the improved prompt:' or explanations."},
                {"role": "user", "content": reflection_prompt}
            ]
            
            response, _ = llm_client.chat_completion(self.model, messages, self.temperature * 0.8, max_tokens=800)
            
            refined_prompt = response.strip()
            
            unwanted_prefixes = [
                "Here is the improved prompt:", "Here's the improved prompt:", "Improved prompt:",
                "REFINED PROMPT:", "Refined prompt:", "Here is the refined prompt:", "The improved prompt is:",
                "**Improved Prompt:**", "**Refined Prompt:**"
            ]
            
            for prefix in unwanted_prefixes:
                if refined_prompt.startswith(prefix):
                    refined_prompt = refined_prompt[len(prefix):].strip()
                    break
            
            if len(refined_prompt) < len(initial_prompt) * 0.5:
                return initial_prompt
            
            return refined_prompt
            
        except Exception as e:
            print(f"Warning: Prompt refinement failed: {e}")
            return initial_prompt
    
    def _get_performance_insights(self, task_type: TaskType, text_analysis: TextAnalysis) -> Dict:
        relevant_history = [
            perf for perf in self.performance_history 
            if perf.task_type == task_type and 
            abs(perf.text_analysis.complexity_score - text_analysis.complexity_score) < 2
        ]
        
        if not relevant_history:
            return {'message': 'No relevant historical data available'}
        
        successful_attempts = [p for p in relevant_history if p.success and p.performance_score > 0.7]
        avg_performance = np.mean([p.performance_score for p in relevant_history])
        
        return {
            'total_attempts': len(relevant_history),
            'successful_attempts': len(successful_attempts),
            'average_performance': avg_performance,
            'success_rate': len(successful_attempts) / len(relevant_history),
            'common_domains': [p.text_analysis.domain_indicators for p in successful_attempts]
        }
    
    def _format_performance_insights(self, insights: Dict) -> str:
        if 'message' in insights:
            return insights['message']
        
        return f"""- Historical attempts: {insights['total_attempts']}
- Success rate: {insights['success_rate']:.1%}
- Average performance: {insights['average_performance']:.2f}
- Best performing contexts: {insights.get('common_domains', 'various')}"""
    
    def _format_additional_context(self, kwargs: Dict) -> str:
        if not kwargs:
            return "None specified"
        
        formatted = []
        for key, value in kwargs.items():
            if isinstance(value, list) and len(value) > 5:
                formatted.append(f"- {key}: {len(value)} items")
            else:
                formatted.append(f"- {key}: {value}")
        
        return '\n'.join(formatted)
    
    def _get_default_prompt(self, task_type: TaskType) -> str:
        defaults = {
            TaskType.CONCEPT_EXTRACTION: "Extract the most important key concepts or entities from the following text. Return a comma-separated list of 1-3 word terms.",
            TaskType.EDGE_INFERENCE: "Analyze the causal relationship between the given concepts based on the text. Determine direction and strength.",
            TaskType.CLUSTERING: "Group the given concepts into semantically related clusters.",
            TaskType.POST_CLUSTERING: "Identify concepts that should be merged based on semantic similarity."
        }
        return defaults.get(task_type, "Analyze the given text and provide relevant insights.")
    
    def record_performance(self, prompt: str, task_type: TaskType, text_analysis: TextAnalysis,
                         performance_score: float, execution_time: float, success: bool,
                         output_quality: float = 0.0, error_message: str = None):
        if not DYNAMIC_PROMPTING_TRACK_PERFORMANCE:
            return
            
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()[:12]
        
        performance_record = PromptPerformance(
            prompt_hash=prompt_hash,
            task_type=task_type,
            text_analysis=text_analysis,
            performance_score=performance_score,
            execution_time=execution_time,
            output_quality=output_quality,
            timestamp=datetime.now(),
            success=success,
            error_message=error_message
        )
        
        self.performance_history.append(performance_record)
        
        if len(self.performance_history) % 50 == 0:
            self._save_performance_history()
            self._cleanup_old_records()
    
    def _save_performance_history(self):
        try:
            history_file = os.path.join(LOG_DIRECTORY, "dynamic_prompt_performance.json")
            os.makedirs(os.path.dirname(history_file), exist_ok=True)
            
            serializable_history = []
            for record in self.performance_history:
                record_dict = asdict(record)
                record_dict['timestamp'] = record.timestamp.isoformat()
                record_dict['task_type'] = record.task_type.value
                serializable_history.append(record_dict)
            
            with open(history_file, 'w') as f:
                json.dump(serializable_history, f, indent=2)
                
        except Exception as e:
            print(f"Warning: Could not save performance history: {e}")
    
    def _load_performance_history(self):
        try:
            history_file = os.path.join(LOG_DIRECTORY, "dynamic_prompt_performance.json")
            if os.path.exists(history_file):
                with open(history_file, 'r') as f:
                    data = json.load(f)
                
                for record_dict in data:
                    record_dict['task_type'] = TaskType(record_dict['task_type'])
                    record_dict['timestamp'] = datetime.fromisoformat(record_dict['timestamp'])
                    record_dict['text_analysis'] = TextAnalysis(**record_dict['text_analysis'])
                    
                    self.performance_history.append(PromptPerformance(**record_dict))
                    
        except Exception as e:
            print(f"Warning: Could not load performance history: {e}")
    
    def _cleanup_old_records(self, max_records: int = 1000):
        if len(self.performance_history) > max_records:
            self.performance_history = sorted(
                self.performance_history,
                key=lambda x: x.timestamp,
                reverse=True
            )[:max_records]
    
    def _setup_logging(self):
        try:
            from config.constants import LOG_DIRECTORY, ENABLE_FILE_LOGGING
            if ENABLE_FILE_LOGGING:
                self.log_file_path = os.path.join(LOG_DIRECTORY, "dynamic_prompting.log")
                os.makedirs(os.path.dirname(self.log_file_path), exist_ok=True)
            else:
                self.log_file_path = None
        except Exception as e:
            print(f"Warning: Could not setup dynamic prompting logging: {e}")
            self.log_file_path = None
    
    def _log_to_file(self, message: str):
        if not self.log_file_path:
            return
            
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(self.log_file_path, 'a', encoding='utf-8') as f:
                f.write(f"[{timestamp}] {message}\n\n")
        except Exception as e:
            print(f"Warning: Could not write to dynamic prompting log: {e}")
    
    def get_adaptation_stats(self) -> Dict:
        if not self.performance_history:
            return {'status': 'no_data'}
        
        recent_records = self.performance_history[-100:]
        
        return {
            'total_prompts_generated': len(self.performance_history),
            'recent_success_rate': np.mean([r.success for r in recent_records]),
            'recent_avg_performance': np.mean([r.performance_score for r in recent_records]),
            'cache_hit_rate': len(self.prompt_cache) / max(len(self.performance_history), 1),
            'domains_covered': len(set(d for r in recent_records for d in r.text_analysis.domain_indicators)),
            'task_distribution': {
                task_type.value: sum(1 for r in recent_records if r.task_type == task_type)
                for task_type in TaskType
            }
        }


def get_dynamic_prompt(task_type: str, text: str, **kwargs) -> str:
    agent = DynamicPromptAgent()
    task_enum = TaskType(task_type)
    prompt, _ = agent.generate_dynamic_prompt(task_enum, text, **kwargs)
    return prompt


def create_global_prompt_agent() -> DynamicPromptAgent:
    return DynamicPromptAgent()


_global_prompt_agent = None

def get_global_prompt_agent() -> DynamicPromptAgent:
    global _global_prompt_agent
    if _global_prompt_agent is None:
        _global_prompt_agent = create_global_prompt_agent()
    return _global_prompt_agent
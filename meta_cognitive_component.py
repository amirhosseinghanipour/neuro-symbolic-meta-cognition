from typing import List, Tuple, Dict, Any, Optional, Set, TYPE_CHECKING
import torch
from datetime import datetime, timedelta
import random
import time

from config import config
from utils import TimeUtils
from symbolic_component import SymbolicReasoner, Fact, Rule
from neural_component import BertExtractor
from integration import NeuralToSymbolicTranslator

if TYPE_CHECKING:
    from rule_generator import DynamicRuleGenerator

class MetaCognitiveEvaluator:
    """
    Evaluates answer plausibility and generates explanations for the reasoning process.
    Includes enhanced plausibility checks.
    """
    MAX_REASONABLE_DURATION_HOURS = 24 * 7
    MAX_REASONABLE_TRAVEL_HOURS = 24

    def evaluate_plausibility(self,
                              answer_fact: Fact,
                              context_facts: Set[Fact]
                             ) -> Tuple[bool, str, float]:
        """
        Checks if the derived answer fact is plausible based on common-sense rules.

        Args:
            answer_fact (Fact): The fact representing the reasoner's final answer.
            context_facts (Set[Fact]): The complete set of facts available during reasoning.

        Returns:
            Tuple[bool, str, float]: (is_plausible, explanation, confidence_score).
        """
        if not isinstance(answer_fact, Fact): return False, "Invalid answer fact.", 0.0
        context_facts = context_facts if isinstance(context_facts, set) else set()

        predicate = answer_fact.predicate; subject = answer_fact.subject; value = answer_fact.object
        explanation = "Evaluation Passed: Answer appears plausible based on implemented checks."
        confidence = 1.0; is_plausible = True

        try:
            if predicate == 'departure_time' and isinstance(value, datetime):
                event_time = next((f.object for f in context_facts if f.predicate == 'event_time' and f.subject == subject and isinstance(f.object, datetime)), None)
                if event_time and value >= event_time:
                    explanation = f"Failed: Departure ({TimeUtils.format_time(value)}) not before Event ({TimeUtils.format_time(event_time)})."
                    is_plausible = False; confidence = 0.0

            elif predicate == 'end_time' and isinstance(value, datetime):
                start_time = next((f.object for f in context_facts if f.predicate == 'start_time' and f.subject == subject and isinstance(f.object, datetime)), None)
                if start_time:
                    if value <= start_time:
                        explanation = f"Failed: End ({TimeUtils.format_time(value)}) not after Start ({TimeUtils.format_time(start_time)})."
                        is_plausible = False; confidence = 0.0
                    else:
                        duration = next((f.object for f in context_facts if f.predicate == 'duration' and f.subject == subject and isinstance(f.object, timedelta)), None)
                        if duration:
                            expected_end = start_time + duration
                            if abs((value - expected_end).total_seconds()) > 60:
                                explanation = f"Warning: End ({TimeUtils.format_time(value)}) differs from Start+Duration ({TimeUtils.format_time(expected_end)})."
                                confidence = 0.7

            elif predicate == 'arrival_time' and isinstance(value, datetime):
                 departure_time = next((f.object for f in context_facts if f.predicate == 'departure_time' and f.subject == subject and isinstance(f.object, datetime)), None)
                 if departure_time:
                      if value <= departure_time:
                           explanation = f"Failed: Arrival ({TimeUtils.format_time(value)}) not after Departure ({TimeUtils.format_time(departure_time)})."
                           is_plausible = False; confidence = 0.0
                      else:
                          travel_time = next((f.object for f in context_facts if f.predicate == 'travel_time' and f.subject == subject and isinstance(f.object, timedelta)), None)
                          if travel_time:
                              expected_arrival = departure_time + travel_time
                              if abs((value - expected_arrival).total_seconds()) > 60:
                                   explanation = f"Warning: Arrival ({TimeUtils.format_time(value)}) differs from Depart+Travel ({TimeUtils.format_time(expected_arrival)})."
                                   confidence = 0.7

            elif predicate == 'duration' and isinstance(value, timedelta):
                if value.total_seconds() < 0:
                    explanation = f"Failed: Duration ({TimeUtils.format_timedelta(value)}) is negative."
                    is_plausible = False; confidence = 0.0
                elif value.total_seconds() > self.MAX_REASONABLE_DURATION_HOURS * 3600:
                     explanation = f"Warning: Duration ({TimeUtils.format_timedelta(value)}) > {self.MAX_REASONABLE_DURATION_HOURS} hours."
                     confidence = 0.6

            elif predicate == 'travel_time' and isinstance(value, timedelta):
                 if value.total_seconds() < 0:
                      explanation = f"Failed: Travel time ({TimeUtils.format_timedelta(value)}) is negative."
                      is_plausible = False; confidence = 0.0
                 elif value.total_seconds() > self.MAX_REASONABLE_TRAVEL_HOURS * 3600:
                      explanation = f"Warning: Travel time ({TimeUtils.format_timedelta(value)}) > {self.MAX_REASONABLE_TRAVEL_HOURS} hours."
                      confidence = 0.6

        except Exception as e:
             explanation = f"Error during plausibility check: {e}"
             is_plausible = False; confidence = 0.0 # Treat errors as implausible

        return is_plausible, explanation, confidence


    def generate_explanation(self,
                             query: Optional[Tuple[str, str, str]],
                             answer_fact: Optional[Fact],
                             reasoning_trace: List[Tuple[Rule, Dict[str, Any], Fact]],
                             plausibility_result: Optional[Tuple[bool, str, float]]
                            ) -> str:
        """
        Generates a human-readable explanation of the reasoning process and evaluation.

        Args:
            query: The query pattern (predicate, subject, object) that was asked.
            answer_fact: The final answer Fact derived by the reasoner, or None.
            reasoning_trace: A list of tuples (Rule, bindings, derived_fact) showing the steps.
            plausibility_result: The tuple returned by `evaluate_plausibility`, or None.

        Returns:
            str: A multi-line string explaining the reasoning and evaluation.
        """
        explanation_lines = []
        query_str = f"{query[0]}({query[1]}, {query[2]})" if query else "No query specified"
        explanation_lines.append(f"Query: {query_str}")

        explanation_lines.append("\nReasoning Steps:")
        if not reasoning_trace:
            status = f"Answer '{answer_fact}' likely initial fact." if answer_fact else "No rules applied or facts missing."
            explanation_lines.append(f" - {status}")
        else:
            relevant_trace = []
            processed_rules = set()
            if answer_fact:
                 final_steps = [step for step in reasoning_trace if step[2] == answer_fact]
                 if final_steps:
                      for step in final_steps:
                           rule_id = step[0].id
                           if (rule_id, step[2]) not in processed_rules:
                                relevant_trace.append(step); processed_rules.add((rule_id, step[2]))
                 # else: explanation_lines.append(f" - Trace does not show derivation of answer {answer_fact}. Showing all steps:")
                 #      relevant_trace = reasoning_trace
            # else: relevant_trace = reasoning_trace # Shows all steps if no answer

            if not relevant_trace and reasoning_trace: relevant_trace = reasoning_trace # Fallback
            relevant_trace.sort(key=lambda step: (step[0].id, str(step[2])))

            if not relevant_trace: explanation_lines.append(" - No relevant reasoning steps found in trace.")
            else:
                 for rule, bindings, derived_fact in relevant_trace:
                     if (rule.id, derived_fact) in processed_rules and len(relevant_trace) > 1 and answer_fact: continue # Avoid re-printing if multiple paths shown
                     processed_rules.add((rule.id, derived_fact))
                     bound_vars_str = {k: (TimeUtils.format_time(v) if isinstance(v, datetime) else (TimeUtils.format_timedelta(v) if isinstance(v, timedelta) else str(v))) for k, v in bindings.items()}
                     explanation_lines.append(f" - Rule {rule.id} ({rule.source}, conf: {rule.confidence:.2f}) with {bound_vars_str} -> {derived_fact}")

        explanation_lines.append(f"\nFinal Answer: {answer_fact if answer_fact else 'Not Found'}")
        explanation_lines.append("\nPlausibility Check:")
        if plausibility_result:
            is_plausible, plaus_explanation, plaus_confidence = plausibility_result
            status = 'Passed' if is_plausible else ('Warning' if plaus_confidence > config.PLAUSIBILITY_THRESHOLD else 'Failed')
            explanation_lines.append(f" - Status: {status} (Conf: {plaus_confidence:.2f}). {plaus_explanation}")
        elif answer_fact: explanation_lines.append(" - Not performed.")
        else: explanation_lines.append(" - Not applicable.")
        return "\n".join(explanation_lines)


class AdaptiveLearner:
    """
    Handles system adaptation based on feedback.

    Adjusts symbolic rule confidences, triggers rule generation, and collects data
    for simulated neural fine-tuning. Runs fine-tuning epochs on collected data.
    """
    def __init__(self,
                 neural_model: BertExtractor,
                 symbolic_reasoner: SymbolicReasoner,
                 rule_generator: 'DynamicRuleGenerator'):
        """
        Initializes the AdaptiveLearner, including optimizer and pseudo-head for fine-tuning.

        Args:
            neural_model (BertExtractor): The neural component instance.
            symbolic_reasoner (SymbolicReasoner): The symbolic component instance.
            rule_generator (DynamicRuleGenerator): The rule generator instance.
        """
        self.neural_model = neural_model
        self.symbolic_reasoner = symbolic_reasoner
        self.rule_generator = rule_generator
        self.feedback_buffer = []

        self.optimizer = None
        self.criterion = None
        self.pseudo_classifier_head = None
        self.num_pseudo_classes = 5 # O, B-TIME, I-TIME, B-DURATION, I-DURATION

        if hasattr(neural_model, 'base_model') and isinstance(neural_model.base_model, torch.nn.Module):
             try:
                 self.optimizer = torch.optim.AdamW(
                     neural_model.base_model.parameters(),
                     lr=config.ADAPTATION_LEARNING_RATE_NEURAL,
                     weight_decay=config.NEURAL_WEIGHT_DECAY
                 )
                 self.criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)
                 bert_hidden_size = neural_model.base_model.config.hidden_size
                 self.pseudo_classifier_head = torch.nn.Linear(bert_hidden_size, self.num_pseudo_classes).to(neural_model.device)
                 self.optimizer.add_param_group({'params': self.pseudo_classifier_head.parameters()})
                 print("INFO: Optimizer, Criterion, and Pseudo-Head initialized for neural fine-tuning simulation.")
             except Exception as e:
                  print(f"WARN: Error initializing optimizer/head for fine-tuning: {e}. Fine-tuning disabled.")
                  self.optimizer = self.criterion = self.pseudo_classifier_head = None
        else:
             print("WARN: Neural model invalid. Neural fine-tuning disabled.")


    def collect_feedback_data(self, question_info: Dict[str, Any], is_correct: bool, answer_fact: Optional[Fact], ground_truth_fact: Optional[Fact]):
         """Stores data needed for a fine-tuning step in a buffer if fine-tuning is enabled."""
         if not self.optimizer: return
         prepared_data = self._prepare_fine_tuning_data(question_info, is_correct, answer_fact, ground_truth_fact)
         if prepared_data: self.feedback_buffer.append(prepared_data)


    def _prepare_fine_tuning_data(self, question_info: Dict[str, Any], is_correct: bool, answer_fact: Optional[Fact], ground_truth_fact: Optional[Fact]) -> Optional[Tuple[Dict, torch.Tensor]]:
        """
        (Simulated) Prepares input tensors and pseudo-target labels for fine-tuning based on feedback.

        Args:
            question_info: Output from BertExtractor.
            is_correct: Whether the symbolic answer was correct.
            answer_fact: The predicted answer fact.
            ground_truth_fact: The correct answer fact.

        Returns:
            Optional[Tuple[Dict, torch.Tensor]]: (inputs, labels) tuple or None if no labels generated.
        """
        text = question_info.get("question")
        current_entities = question_info.get("entities", {})
        if not text: return None

        try:
            inputs_with_offset = self.neural_model.base_tokenizer(
                text, return_tensors='pt', max_length=config.MAX_LENGTH,
                truncation=True, padding='max_length', return_offsets_mapping=True
            )
        except Exception as e: return None
        offset_mapping = inputs_with_offset["offset_mapping"].squeeze(0).tolist()
        input_ids = inputs_with_offset["input_ids"].squeeze(0)
        inputs = {k: v for k, v in inputs_with_offset.items() if k != 'offset_mapping'}
        pseudo_labels = torch.full_like(input_ids, -100)
        label_generated = False

        if not is_correct and ground_truth_fact:
            gt_type_label, gt_type_label_i = -1, -1; possible_gt_texts = []
            if isinstance(ground_truth_fact.object, datetime): gt_type_label, gt_type_label_i = 1, 2; time_str = TimeUtils.format_time(ground_truth_fact.object); possible_gt_texts = [time_str, time_str.replace(" AM","").replace(" PM","")]
            elif isinstance(ground_truth_fact.object, timedelta): gt_type_label, gt_type_label_i = 3, 4; possible_gt_texts = [TimeUtils.format_timedelta(ground_truth_fact.object)]

            if gt_type_label != -1:
                 found_gt_span = None
                 for gt_str in possible_gt_texts:
                      if not gt_str or "Invalid" in gt_str: continue
                      try:
                           match_start = text.find(gt_str)
                           if match_start != -1: found_gt_span = (match_start, match_start + len(gt_str)); break
                      except: pass
                 if found_gt_span:
                      start_char, end_char = found_gt_span; first = True
                      for i, (off_s, off_e) in enumerate(offset_mapping):
                           if off_s == 0 and off_e == 0: continue
                           if max(start_char, off_s) < min(end_char, off_e):
                                pseudo_labels[i] = gt_type_label if first else gt_type_label_i; first = False; label_generated = True

        elif is_correct and answer_fact:
             entities_to_reinforce = current_entities.get('times', []) + current_entities.get('durations', [])
             for entity_data in entities_to_reinforce:
                  start_char, end_char = entity_data['start'], entity_data['end']; entity_val = entity_data['value']
                  pseudo_cls, pseudo_cls_i = -1, -1
                  if isinstance(entity_val, datetime): pseudo_cls, pseudo_cls_i = 1, 2
                  elif isinstance(entity_val, timedelta): pseudo_cls, pseudo_cls_i = 3, 4
                  if pseudo_cls != -1:
                       first = True
                       for i, (off_s, off_e) in enumerate(offset_mapping):
                            if off_s == 0 and off_e == 0: continue
                            if max(start_char, off_s) < min(end_char, off_e):
                                 pseudo_labels[i] = pseudo_cls if first else pseudo_cls_i; first = False; label_generated = True

        if label_generated:
            inputs = {k: v.to(self.neural_model.device) for k, v in inputs.items()}
            pseudo_labels = pseudo_labels.to(self.neural_model.device)
            return inputs, pseudo_labels
        else: return None


    def fine_tune_neural_step(self, batch_inputs: List[Dict[str, torch.Tensor]], batch_labels: List[torch.Tensor]) -> bool:
        """
        Performs a simulated gradient update step on a batch of feedback data.

        Args:
            batch_inputs (List[Dict[str, torch.Tensor]]): List of input dicts for the batch.
            batch_labels (List[torch.Tensor]): List of label tensors for the batch.

        Returns:
            bool: True if the step was performed successfully, False otherwise.
        """
        if not self.optimizer or not self.criterion or not self.pseudo_classifier_head or not hasattr(self.neural_model, 'base_model'): return False
        model = self.neural_model.base_model; head = self.pseudo_classifier_head
        model.train(); head.train(); self.optimizer.zero_grad()
        try:
             input_ids = torch.cat([d['input_ids'] for d in batch_inputs], dim=0)
             attention_mask = torch.cat([d['attention_mask'] for d in batch_inputs], dim=0)
             labels = torch.stack(batch_labels, dim=0) # Use stack as labels are [SeqLen]
        except Exception as e: print(f"Error preparing batch tensors: {e}"); model.eval(); head.eval(); return False
        if input_ids.shape[0] == 0: model.eval(); head.eval(); return False

        try:
             outputs = model(input_ids=input_ids, attention_mask=attention_mask)
             sequence_output = outputs.last_hidden_state
             logits = head(sequence_output)
             loss = self.criterion(logits.view(-1, self.num_pseudo_classes), labels.view(-1))
        except Exception as e: print(f"ERROR during neural forward/loss: {e}"); model.eval(); head.eval(); return False
        try:
            loss.backward(); self.optimizer.step()
        except Exception as e: print(f"ERROR during neural backward/step: {e}"); model.eval(); head.eval(); return False
        model.eval(); head.eval()
        return True


    def run_fine_tuning_epoch(self):
        """Runs simulated fine-tuning steps over the collected feedback data buffer."""
        if not self.feedback_buffer or not self.optimizer:
            print("INFO: Fine-tuning epoch skipped (no data or optimizer unavailable).")
            self.feedback_buffer.clear(); return

        print(f"--- Running Neural Fine-Tuning Epoch on {len(self.feedback_buffer)} feedback points ---")
        start_time = time.time(); steps_run = 0; num_batches = 0
        random.shuffle(self.feedback_buffer)

        for i in range(0, len(self.feedback_buffer), config.NEURAL_TUNING_BATCH_SIZE):
            batch = self.feedback_buffer[i : i + config.NEURAL_TUNING_BATCH_SIZE]
            batch_inputs = [item[0] for item in batch]
            batch_labels = [item[1] for item in batch]
            if not batch_inputs or not batch_labels: continue # Skip empty batches
            num_batches += 1
            step_success = self.fine_tune_neural_step(batch_inputs, batch_labels)
            if step_success: steps_run += 1

        end_time = time.time()
        print(f"--- Fine-Tuning Epoch Finished ({steps_run}/{num_batches} batches successful) ---")
        print(f"  Time taken: {end_time - start_time:.2f}s")
        self.feedback_buffer.clear()


    def update_on_feedback(self,
                           question_info: Dict[str, Any],
                           answer_fact: Optional[Fact],
                           reasoning_trace: List[Tuple[Rule, Dict[str, Any], Fact]],
                           is_correct: bool,
                           ground_truth_fact: Optional[Fact] = None):
        """
        Updates symbolic components and collects data for neural fine-tuning based on feedback.

        Args:
            question_info: Output from BertExtractor.
            answer_fact: The predicted answer fact.
            reasoning_trace: List of (Rule, bindings, derived_fact) tuples.
            is_correct: Whether the prediction matches the ground truth.
            ground_truth_fact: The correct Fact object (optional).
        """
        print(f"\n--- Adaptation Triggered ---")
        print(f"  Question: \"{question_info.get('question', 'N/A')[:50]}...\"")
        print(f"  Answer Correct: {is_correct}")
        print(f"  Provided Answer: {answer_fact}")

        relevant_rule_ids = set()
        if answer_fact and reasoning_trace:
             final_steps = [step for step in reasoning_trace if step[2] == answer_fact]
             for step in final_steps: relevant_rule_ids.add(step[0].id)

        if is_correct:
            if answer_fact and relevant_rule_ids:
                print("  Action: Strengthening symbolic rules.")
                for rule in self.symbolic_reasoner.rules:
                    if rule.id in relevant_rule_ids:
                        delta = config.ADAPTATION_LEARNING_RATE_SYMBOLIC * (1.0 - rule.confidence)
                        rule.confidence = min(1.0, rule.confidence + delta)
        else:
            if answer_fact:
                print("  Action: Weakening symbolic rules.")
                if relevant_rule_ids:
                    for rule in self.symbolic_reasoner.rules:
                        if rule.id in relevant_rule_ids:
                            delta = config.ADAPTATION_LEARNING_RATE_SYMBOLIC * rule.confidence
                            rule.confidence = max(0.05, rule.confidence - delta)
            else:
                print("  Action: Attempting to address missing answer (Symbolic).")
                query = question_info.get('query'); relations = question_info.get('relations', [])
                if query:
                     query_pred = query[0]
                     needs_time = any(p in query_pred for p in ['time', 'when', 'arrive', 'depart', 'end', 'start'])
                     needs_duration = any(p in query_pred for p in ['duration', 'long', 'takes', 'travel'])
                     has_time = any(isinstance(r[2], datetime) for r in relations if isinstance(r, tuple) and len(r)>2)
                     has_duration = any(isinstance(r[2], timedelta) for r in relations if isinstance(r, tuple) and len(r)>2)
                     if needs_time and not has_time: print("    - Possible Issue: Query requires time, but none reliably extracted.")
                     if needs_duration and not has_duration: print("    - Possible Issue: Query requires duration, but none reliably extracted.")
                else: print("    - Possible Issue: Query itself was not identified by neural component.")

                print("  Action: Triggering dynamic rule generation.")
                temp_reasoner = SymbolicReasoner(); temp_translator = NeuralToSymbolicTranslator()
                if question_info: temp_translator.translate(question_info, temp_reasoner)
                initial_facts = temp_reasoner.get_all_facts()
                if hasattr(self, 'rule_generator') and self.rule_generator:
                    new_rule = self.rule_generator.generate_rule_from_failure(question_info, initial_facts, self.symbolic_reasoner.get_all_rules(), ground_truth_fact)
                    if new_rule:
                        print(f"    - Generated new rule: {new_rule}")
                        if not self.rule_generator._is_rule_redundant(new_rule, self.symbolic_reasoner.get_all_rules()):
                             added_rule_obj = self.symbolic_reasoner.add_rule(conditions=new_rule.conditions, conclusion=new_rule.conclusion, confidence=new_rule.confidence, source=new_rule.source)
                             print(f"    - Added Rule {added_rule_obj.id} to the reasoner.")
                        else: print(f"    - Generated rule is redundant with existing rules. Not added.")
                    else:
                        print("    - Rule generation did not produce a new rule for this specific case.")
                        self.rule_generator.store_failure(question_info, initial_facts, ground_truth_fact)
                else: print("   - ERROR: Rule generator not available in AdaptiveLearner.")

        self.collect_feedback_data(question_info, is_correct, answer_fact, ground_truth_fact)

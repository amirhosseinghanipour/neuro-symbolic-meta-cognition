from typing import List, Dict, Any, Optional, Set, Callable
import time
import random
from datetime import datetime, timedelta

from config import config
from utils import TimeUtils
from symbolic_component import Fact

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from meta_cognitive_component import AdaptiveLearner
    from main import NeuroSymbolicPipeline

def evaluate_answer(predicted_fact: Optional[Fact],
                    ground_truth_pred: str,
                    ground_truth_val_str: str,
                    parsed_ground_truth_val: Any) -> bool:
    """
    Compares the predicted answer fact with the ground truth information.

    Handles type conversions and comparisons for time and duration values.

    Args:
        predicted_fact (Optional[Fact]): The predicted Fact object or None.
        ground_truth_pred (str): The expected predicate string.
        ground_truth_val_str (str): The expected answer value string.
        parsed_ground_truth_val (Any): The pre-parsed ground truth value.

    Returns:
        bool: True if the prediction matches the ground truth, False otherwise.
    """
    if predicted_fact is None: return parsed_ground_truth_val is None
    if not isinstance(predicted_fact, Fact): return False
    if predicted_fact.predicate != ground_truth_pred: return False
    pred_val = predicted_fact.object; gt_val = parsed_ground_truth_val
    if gt_val is None and ground_truth_val_str is not None: return str(pred_val) == ground_truth_val_str
    try:
        if isinstance(pred_val, datetime) and isinstance(gt_val, datetime): return pred_val.strftime("%H:%M") == gt_val.strftime("%H:%M")
        elif isinstance(pred_val, timedelta) and isinstance(gt_val, timedelta): return abs(pred_val.total_seconds() - gt_val.total_seconds()) < 60
        elif type(pred_val) == type(gt_val): return pred_val == gt_val
        else: return str(pred_val) == ground_truth_val_str
    except Exception as e: print(f"Error during value comparison: {e}"); return False


def run_evaluation_epochs(dataset: List[Dict[str, Any]],
                          pipeline_instance: 'NeuroSymbolicPipeline',
                          num_epochs: int = config.NUM_EPOCHS
                         ) -> List[Dict[str, float]]:
    """
    Runs the full pipeline over multiple epochs, performs evaluation, and triggers adaptation.

    Collects feedback during each epoch and runs neural fine-tuning and batch rule generation
    at the end of each epoch based on configuration.

    Args:
        dataset (List[Dict[str, Any]]): List of QA dictionaries from load_qa_data.
        pipeline_instance (NeuroSymbolicPipeline): An initialized pipeline instance.
        num_epochs (int): The number of passes over the dataset.

    Returns:
        List[Dict[str, float]]: A list of metric dictionaries, one for each epoch.
    """
    epoch_metrics = []
    total_start_time = time.time()
    if not dataset: print("Evaluation dataset is empty."); return []

    adapter = getattr(pipeline_instance, 'adapter', None)
    if not adapter: print("ERROR: Pipeline instance lacks 'adapter'. Cannot run adaptation."); return []

    def pipeline_func_wrapper(question: str) -> Dict[str, Any]:
         try:
             result = pipeline_instance.process_question(question)
             result['adapter'] = adapter
             return result
         except Exception as e:
             print(f"ERROR in pipeline_func_wrapper for '{question[:50]}...': {e}")
             return {"question": question, "answer_fact": None, "adapter": adapter, "question_info": None, "reasoning_trace": [], "explanation": f"Pipeline failed: {e}"}

    for epoch in range(num_epochs):
        print(f"\n===== Starting Epoch {epoch + 1}/{num_epochs} =====")
        epoch_start_time = time.time(); correct_count = 0; processed_count = 0
        total_processing_time_epoch = 0; pipeline_failures_epoch = 0
        adapter.feedback_buffer.clear()

        for i, item in enumerate(dataset):
            question = item.get('question'); gt_pred = item.get('answer_predicate')
            gt_val_str = item.get('answer_value'); parsed_gt_val = item.get('parsed_answer_value')
            item_id = item.get('id', f'item_{i}')
            if not all([isinstance(question, str), isinstance(gt_pred, str), gt_val_str is not None]): continue

            item_start_time = time.time(); is_correct = False; pipeline_failed = False; result = {}
            try: result = pipeline_func_wrapper(question)
            except Exception as e: print(f"  ERROR: Pipeline execution failed for {item_id}: {e}"); pipeline_failed = True
            item_end_time = time.time(); processing_time = (item_end_time - item_start_time) * 1000

            if not pipeline_failed:
                processed_count += 1; total_processing_time_epoch += processing_time
                predicted_fact = result.get('answer_fact'); question_info = result.get('question_info')
                reasoning_trace = result.get('reasoning_trace', [])
                is_correct = evaluate_answer(predicted_fact, gt_pred, gt_val_str, parsed_gt_val)
                if is_correct: correct_count += 1

                if adapter and question_info:
                     ground_truth_fact_obj = None
                     if gt_pred and parsed_gt_val is not None:
                          subject = query[1] if (query := question_info.get('query')) and isinstance(query, tuple) and len(query) > 1 else 'event_gt'
                          try: ground_truth_fact_obj = Fact(predicate=gt_pred, subject=subject, object=parsed_gt_val)
                          except: pass # Ignore errors creating GT fact
                     try:
                          adapter.update_on_feedback(question_info, predicted_fact, reasoning_trace, is_correct, ground_truth_fact_obj)
                     except Exception as e: print(f"  ERROR: Adaptation step failed for {item_id}: {e}")
            else: pipeline_failures_epoch += 1

        epoch_end_time = time.time()
        accuracy = correct_count / processed_count if processed_count > 0 else 0.0
        average_time = total_processing_time_epoch / processed_count if processed_count > 0 else 0.0
        print(f"\n--- Epoch {epoch + 1} Summary ---")
        print(f"  Processed: {processed_count}/{len(dataset)} (Failures: {pipeline_failures_epoch})")
        print(f"  Correct: {correct_count}; Accuracy: {accuracy:.4f}")
        print(f"  Avg Time/Q: {average_time:.2f} ms; Epoch Duration: {epoch_end_time - epoch_start_time:.2f}s")
        metrics = {"epoch": epoch + 1, "accuracy": accuracy, "average_time_ms": average_time, "processed_count": processed_count, "failures": pipeline_failures_epoch}
        epoch_metrics.append(metrics)

        if config.RUN_FINE_TUNING_AFTER_EPOCH and adapter: adapter.run_fine_tuning_epoch()

        print("\n--- Attempting Post-Epoch Batch Rule Generation ---")
        if hasattr(pipeline_instance, 'rule_generator') and hasattr(pipeline_instance, 'reasoner'):
            batch_rules = pipeline_instance.rule_generator.generate_rules_from_stored_failures(pipeline_instance.reasoner.get_all_rules())
            if batch_rules:
                added_count = 0
                for rule in batch_rules:
                    if not pipeline_instance.rule_generator._is_rule_redundant(rule, pipeline_instance.reasoner.get_all_rules()):
                         added_rule = pipeline_instance.reasoner.add_rule(conditions=rule.conditions, conclusion=rule.conclusion, confidence=rule.confidence, source=rule.source)
                         print(f"   Added Batch Rule {added_rule.id} to reasoner.")
                         added_count += 1
                print(f"Added {added_count} unique rules from batch generation.")
            else: print("No new rules generated from batch processing.")
        else: print("WARN: Rule generator/reasoner not found for batch generation.")

    total_end_time = time.time()
    print(f"\n===== Evaluation Finished ({num_epochs} Epochs); Total Time: {total_end_time - total_start_time:.2f}s =====")
    return epoch_metrics

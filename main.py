import time
import os
import pprint
import traceback
from typing import Dict, Any, List

from config import config
from utils import TimeUtils
from data_loader import load_qa_data, create_sample_data_file
from evaluation import run_evaluation_epochs
from neural_component import BertExtractor
from symbolic_component import SymbolicReasoner, Fact, Rule
from integration import NeuralToSymbolicTranslator
from meta_cognitive_component import MetaCognitiveEvaluator, AdaptiveLearner
from rule_generator import DynamicRuleGenerator

class NeuroSymbolicPipeline:
    """
    Orchestrates the Neuro-Symbolic process with Meta-Cognition.

    Integrates neural extraction, symbolic translation, reasoning, meta-cognitive
    evaluation, explanation generation, and adaptation components.
    """
    def __init__(self):
        """
        Initializes all components of the pipeline.

        Raises:
            RuntimeError: If any component fails to initialize.
        """
        print("--- Initializing Neuro-Symbolic Pipeline ---")
        start_init = time.time()
        try:
            self.extractor = BertExtractor(base_model_name=config.BASE_BERT_MODEL, ner_model_name=config.NER_MODEL, device=config.DEVICE)
            self.reasoner = SymbolicReasoner()
            self.translator = NeuralToSymbolicTranslator()
            self.evaluator = MetaCognitiveEvaluator()
            self.rule_generator = DynamicRuleGenerator()
            self.adapter = AdaptiveLearner(self.extractor, self.reasoner, self.rule_generator)
        except Exception as e:
             print(f"FATAL ERROR during pipeline initialization: {e}")
             traceback.print_exc()
             raise RuntimeError("Pipeline initialization failed") from e
        end_init = time.time()
        print(f"--- Pipeline Initialization Complete ({end_init - start_init:.2f}s) ---")

    def process_question(self, question: str) -> Dict[str, Any]:
        """
        Processes a single question through the entire neuro-symbolic pipeline.

        Includes extraction, translation, reasoning, evaluation, and explanation.

        Args:
            question (str): The input natural language question string.

        Returns:
            Dict[str, Any]: A dictionary containing detailed results including:
                            'question', 'question_info', 'initial_facts', 'derived_facts',
                            'all_facts', 'reasoning_trace', 'answer_fact', 'plausibility',
                            'explanation', 'adapter'.
        """
        result_dict = {
            "question": question, "question_info": None, "initial_facts": set(),
            "derived_facts": set(), "all_facts": set(), "reasoning_trace": [],
            "answer_fact": None, "plausibility": None, "explanation": "Processing failed early.",
            "adapter": self.adapter
        }
        try:
            neural_output = self.extractor.extract_entities_relations(question)
            result_dict["question_info"] = neural_output
            query_pattern = neural_output.get('query')

            self.reasoner.clear_facts()
            num_translated = self.translator.translate(neural_output, self.reasoner)
            initial_facts = self.reasoner.get_all_facts().copy()
            result_dict["initial_facts"] = initial_facts

            derived_facts, reasoning_trace = self.reasoner.reason()
            all_facts = self.reasoner.get_all_facts()
            result_dict["derived_facts"] = derived_facts
            result_dict["reasoning_trace"] = reasoning_trace
            result_dict["all_facts"] = all_facts

            answer_fact = None
            if query_pattern:
                results = self.reasoner.query(query_pattern)
                if results: answer_fact = results[0]
            result_dict["answer_fact"] = answer_fact

            plausibility_result = None
            if answer_fact:
                plausibility_result = self.evaluator.evaluate_plausibility(answer_fact, all_facts)
            result_dict["plausibility"] = plausibility_result

            explanation = self.evaluator.generate_explanation(query_pattern, answer_fact, reasoning_trace, plausibility_result)
            result_dict["explanation"] = explanation
        except Exception as e:
             print(f"ERROR during pipeline processing for question '{question[:50]}...': {e}")
             traceback.print_exc()
             result_dict["explanation"] = f"Pipeline processing failed: {e}"
        return result_dict


def run_main_experiment():
    """Main function to initialize the pipeline, load data, and run evaluation epochs."""
    print("=============================================")
    print("=== Neuro-Symbolic Meta-Cognitive System ===")
    print("=============================================")

    try:
        pipeline = NeuroSymbolicPipeline()
    except Exception as e:
        print(f"Could not initialize pipeline: {e}. Exiting.")
        return

    create_sample_data_file(config.DATA_PATH)
    dataset = load_qa_data(config.DATA_PATH)

    if dataset and hasattr(pipeline, 'adapter'):
        print("\n--- Running Evaluation & Adaptation Epochs ---")
        all_epoch_metrics = run_evaluation_epochs(dataset, pipeline, num_epochs=config.NUM_EPOCHS)
        print(f"\n--- Final Evaluation Metrics (Epochs) ---")
        pprint.pprint(all_epoch_metrics)
    else:
        print("Evaluation skipped: Failed to load dataset or pipeline invalid.")

    print("\n=============================================")
    print("=== Processing New Question Post-Adaptation ===")
    print("=============================================")
    new_q = "My train journey starts at 1 PM and takes 90 minutes. What time do I arrive?"
    try:
        result = pipeline.process_question(new_q)
        print(f"\nProcessed New Question: '{new_q}'")
        print(f"Predicted Answer: {result.get('answer_fact')}")
        print("\n--- Explanation for New Question ---")
        print(result.get('explanation', 'No explanation available.'))
    except Exception as e:
        print(f"ERROR processing new question '{new_q}': {e}")
        traceback.print_exc()

    print("\n--- Final State of Rules ---")
    if hasattr(pipeline, 'reasoner'):
        rules = sorted(pipeline.reasoner.get_all_rules(), key=lambda r: r.id)
        if rules: [print(rule) for rule in rules]
        else: print("No rules found in the reasoner.")
    else: print("Pipeline or reasoner object not available.")

    print("\n=============================================")
    print("=== Execution Finished ===")
    print("=============================================")

if __name__ == "__main__":
    run_main_experiment()

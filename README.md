# Neuro-Symbolic Temporal QA Pipeline

This document provides a low-level overview of the Neuro-Symbolic Temporal Question Answering system based on the provided Python code snippets (structured as notebook sections/potential `.py` files). It focuses on the code components, setup, and execution flow.

## 1. Core Description

The project implements a pipeline that combines neural language processing (BERT) with symbolic reasoning to answer questions primarily focused on temporal aspects (time, duration, scheduling). It includes meta-cognitive capabilities like plausibility checking, explanation generation, and adaptive learning through rule confidence adjustment and dynamic rule generation.

## 2. Core Components & Modules

The system is modular, with components corresponding roughly to the provided code sections:

*   **`config.py` (Section 0):** Holds configuration parameters like BERT model name, data paths, reasoning depth, learning rates, thresholds, etc.
*   **`utils.py` (Section 1):** Contains utility functions, notably `TimeUtils` for parsing and formatting time/duration strings.
*   **`neural_component.py` (Section 2):**
    *   `BertExtractor`: Loads a pre-trained BERT model (`transformers` library). Provides sentence embeddings (`get_embeddings`) and performs *simplified* entity/relation extraction using regex/keywords (`extract_entities_relations`). **Note:** NER/RE is a placeholder for now.
*   **`symbolic_component.py` (Section 3):**
    *   `Fact`: Dataclass representing grounded facts (predicate, subject, object). Immutable and hashable.
    *   `Rule`: Dataclass representing IF-THEN rules with conditions, conclusion, confidence, source, and ID.
    *   `SymbolicReasoner`: Implements a forward-chaining reasoner. Manages `Fact`s and `Rule`s. Includes core temporal rules. Methods: `add_fact`, `add_rule`, `reason`, `query`, `clear_facts`, etc.
*   **`integration.py` (Section 4):**
    *   `NeuralToSymbolicTranslator`: Translates the dictionary output from `BertExtractor` into `Fact` objects suitable for the `SymbolicReasoner`.
*   **`meta_cognitive_component.py` (Section 5):**
    *   `MetaCognitiveEvaluator`: Checks answer plausibility (`evaluate_plausibility`) based on simple temporal constraints and generates step-by-step explanations (`generate_explanation`).
    *   `AdaptiveLearner`: Receives feedback (`is_correct`) and adapts the system (`update_on_feedback`). Adjusts rule confidences or triggers the `DynamicRuleGenerator`.
*   **`rule_generator.py` (Section 6):**
    *   `DynamicRuleGenerator`: Attempts to generate new `Rule` objects when reasoning fails. Uses predefined templates (`_apply_templates`) and stores failures (`store_failure`). Includes experimental batch generation based on clustering embeddings (`generate_rules_from_stored_failures`). Checks for redundancy (`_is_rule_redundant`).
*   **`data_loader.py` (Section 7):**
    *   `load_qa_data`: Loads QA pairs from a JSON file, parsing answer values.
    *   `create_sample_data_file`: Creates a default sample JSON data file if none exists.
*   **`evaluation.py` (Section 8):**
    *   `evaluate_answer`: Compares a predicted `Fact` against ground truth.
    *   `run_evaluation`: Iterates through a dataset, runs the pipeline via a passed function, evaluates, triggers adaptation (`AdaptiveLearner.update_on_feedback`), and reports metrics.
*   **`main.py` (Section 9):**
    *   `NeuroSymbolicPipeline`: Orchestrates the entire process by initializing and connecting all components. Provides the main `process_question` method.
    *   Main execution block (`if __name__ == "__main__":`): Initializes the pipeline, loads data, runs the evaluation loop (`run_pipeline_evaluation_main`), and optionally processes a final standalone question.

## 3. Execution Flow

1.  Run the main script (e.g., `python main.py` if structured as files, or execute the final cell in the notebook).
2.  `main()` function executes:
    *   Initializes `NeuroSymbolicPipeline`, which in turn initializes all components (`BertExtractor`, `SymbolicReasoner`, `NeuralToSymbolicTranslator`, `MetaCognitiveEvaluator`, `DynamicRuleGenerator`, `AdaptiveLearner`). This includes loading the BERT model.
    *   Calls `run_pipeline_evaluation_main`.
        *   This function ensures the sample data file exists (`create_sample_data_file`).
        *   Loads the dataset using `load_qa_data`.
        *   Calls `run_evaluation` from `evaluation.py`, passing the dataset and a wrapper around `pipeline.process_question`.
        *   `run_evaluation` iterates through each question in the dataset:
            *   Calls `pipeline.process_question`.
                *   `BertExtractor` processes the question (embedding, basic NER/RE, query ID).
                *   `NeuralToSymbolicTranslator` converts neural output to initial `Fact`s in the `SymbolicReasoner`.
                *   `SymbolicReasoner.reason()` applies rules to derive new `Fact`s.
                *   The answer `Fact` is retrieved using `SymbolicReasoner.query`.
                *   `MetaCognitiveEvaluator.evaluate_plausibility` checks the answer.
                *   `MetaCognitiveEvaluator.generate_explanation` creates the explanation string.
            *   `evaluate_answer` compares the predicted answer fact to the ground truth.
            *   `AdaptiveLearner.update_on_feedback` is called with the correctness result, potentially adjusting rule confidences or triggering/storing for `DynamicRuleGenerator`.
        *   `run_evaluation` prints final metrics (accuracy, avg. time).
        *   `run_pipeline_evaluation_main` attempts post-evaluation batch rule generation using `DynamicRuleGenerator.generate_rules_from_stored_failures`.
    *   After evaluation, `main()` processes a sample standalone question using `pipeline.process_question` to demonstrate the pipeline's state post-adaptation.
    *   Finally, it prints the final set of rules in the reasoner.

## 4. Configuration

Key parameters are set in `config.py`. These include:

*   `BERT_MODEL_NAME`: Which pre-trained BERT model to use.
*   `DEVICE`: Compute device (`cpu` or `cuda`).
*   `DATA_PATH`: Path to the QA dataset JSON file.
*   `MAX_LENGTH`: Max sequence length for BERT tokenizer.
*   `MAX_REASONING_DEPTH`: Limit for the symbolic reasoner's iterations.
*   `INITIAL_RULE_CONFIDENCE`: Default confidence for new rules.
*   `ADAPTATION_LEARNING_RATE_SYMBOLIC`: Rate for adjusting rule confidences.
*   `PLAUSIBILITY_THRESHOLD`: Confidence threshold used in explanations.
*   `RULE_GEN_MIN_SUPPORT`: Minimum cluster size/failures needed for batch rule generation.
*   `RULE_GEN_CLUSTER_THRESHOLD`: Cosine similarity threshold for clustering failures.

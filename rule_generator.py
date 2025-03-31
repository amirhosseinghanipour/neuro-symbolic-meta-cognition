import random
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from typing import List, Set, Dict, Optional, Tuple
from collections import Counter
import re
import torch

from config import config
from utils import TimeUtils
from symbolic_component import Fact, Rule

class DynamicRuleGenerator:
    """
    Generates new symbolic rules based on reasoning failures using templates and clustering.

    Uses template matching for immediate rule generation attempts and stores failures
    for batch processing using embedding clustering to find common failure patterns.
    """

    def __init__(self):
        """Initializes the rule generator with rule templates and failure store."""
        self.templates = [
            {'conditions': [('event_time', '?e', '?t'), ('duration', '?e', '?d')], 'conclusion': ('start_work_time', '?e', 'calculate_departure(?t, ?d)'), 'required_query': "start_work_time"},
            {'conditions': [('start_time', '?e', '?t1'), ('end_time', '?e', '?t2')], 'conclusion': ('duration', '?e', 'calculate_duration(?t1, ?t2)'), 'required_query': "duration"},
            {'conditions': [('departure_time', '?e', '?t'), ('travel_time', '?e', '?d')], 'conclusion': ('arrival_time', '?e', 'calculate_end(?t, ?d)'), 'required_query': "arrival_time"},
            {'conditions': [('start_time', '?e', '?t')], 'conclusion': ('event_time', '?e', '?t'), 'required_query': "event_time"},
            {'conditions': [('start_time', '?e', '?t'), ('travel_time', '?e', '?d')], 'conclusion': ('arrival_time', '?e', 'calculate_end(?t, ?d)'), 'required_query': "arrival_time"},
            {'conditions': [('event_time', '?e', '?t'), ('travel_time', '?e', '?d')], 'conclusion': ('departure_time', '?e', 'calculate_departure(?t, ?d)'), 'required_query': "departure_time"},
            {'conditions': [('start_time', '?e', '?t'), ('duration', '?e', '?d')], 'conclusion': ('end_time', '?e', 'calculate_end(?t, ?d)'), 'required_query': "end_time"},
        ]
        self.failed_cases_store: List[Dict[str, Any]] = []
        print(f"Initialized DynamicRuleGenerator with {len(self.templates)} templates.")

    def store_failure(self, question_info: Dict, initial_facts: Set[Fact], ground_truth: Optional[Fact] = None):
         """
         Stores information about a reasoning failure for potential batch processing.

         Args:
             question_info (Dict): Output from BertExtractor.
             initial_facts (Set[Fact]): Facts available before reasoning.
             ground_truth (Optional[Fact]): Correct answer fact, if known.
         """
         if not isinstance(question_info, dict) or not isinstance(initial_facts, set): return
         self.failed_cases_store.append({
             "question_info": question_info, "initial_facts": initial_facts, "ground_truth": ground_truth
         })

    def _is_rule_redundant(self, new_rule: Rule, existing_rules: List[Rule]) -> bool:
        """
        Checks if a newly generated rule is functionally redundant with existing rules.

        Compares conclusion predicate, condition predicates, and calculation function.

        Args:
            new_rule (Rule): The candidate rule.
            existing_rules (List[Rule]): List of rules already present.

        Returns:
            bool: True if a redundant rule is found, False otherwise.
        """
        new_rule_cond_preds = set(c[0] for c in new_rule.conditions)
        new_rule_conc_pred = new_rule.conclusion[0]
        new_rule_conc_calc = None
        if isinstance(new_rule.conclusion[2], str) and new_rule.conclusion[2].startswith("calculate_"):
             match = re.match(r'(calculate_\w+)\(.*\)', new_rule.conclusion[2])
             if match: new_rule_conc_calc = match.group(1)
        for rule in existing_rules:
            if rule.conclusion[0] == new_rule_conc_pred:
                existing_rule_cond_preds = set(c[0] for c in rule.conditions)
                if existing_rule_cond_preds == new_rule_cond_preds:
                     existing_rule_conc_calc = None
                     if isinstance(rule.conclusion[2], str) and rule.conclusion[2].startswith("calculate_"):
                          match = re.match(r'(calculate_\w+)\(.*\)', rule.conclusion[2])
                          if match: existing_rule_conc_calc = match.group(1)
                     if new_rule_conc_calc == existing_rule_conc_calc: return True
        return False

    def _apply_templates(self, query_predicate: str, sample_facts: Set[Fact], existing_rules: List[Rule]) -> Optional[Rule]:
         """
         Tries to instantiate a rule template based on the query and available facts.

         Args:
             query_predicate (str): The predicate of the fact the system failed to derive.
             sample_facts (Set[Fact]): Facts available in the failed case.
             existing_rules (List[Rule]): Current rules for redundancy check.

         Returns:
             Optional[Rule]: A new Rule object if a suitable, non-redundant template is found, else None.
         """
         available_fact_predicates = {f.predicate for f in sample_facts}
         for template in self.templates:
            if template["required_query"] == query_predicate:
                required_cond_preds = {cond[0] for cond in template["conditions"]}
                if required_cond_preds.issubset(available_fact_predicates):
                    new_rule = Rule(conditions=template["conditions"], conclusion=template["conclusion"],
                                    confidence=max(0.1, config.INITIAL_RULE_CONFIDENCE * 0.8), source="generated")
                    if not self._is_rule_redundant(new_rule, existing_rules): return new_rule
         return None

    def generate_rule_from_failure(self, question_info: Dict, initial_facts: Set[Fact], existing_rules: List[Rule], ground_truth_fact: Optional[Fact] = None) -> Optional[Rule]:
        """
        Attempts to generate a single rule immediately following a reasoning failure using templates.

        Args:
            question_info (Dict): Output from BertExtractor.
            initial_facts (Set[Fact]): Facts available before reasoning failed.
            existing_rules (List[Rule]): Current rules in the reasoner.
            ground_truth_fact (Optional[Fact]): The correct answer fact (currently unused here).

        Returns:
            Optional[Rule]: A new Rule object if successful and non-redundant, otherwise None.
        """
        query = question_info.get("query")
        if not query or not isinstance(query, tuple) or len(query) != 3:
            print("WARN: Rule generation skipped: Invalid or missing query.")
            return None
        query_predicate = query[0]
        generated_rule = self._apply_templates(query_predicate, initial_facts, existing_rules)
        return generated_rule


    def generate_rules_from_stored_failures(self, existing_rules: List[Rule]) -> List[Rule]:
        """
        Attempts to generate rules by analyzing clusters of similar stored failures using embeddings.

        Args:
            existing_rules (List[Rule]): Current rules for redundancy checks.

        Returns:
            List[Rule]: A list of newly generated, non-redundant Rule objects.
        """
        newly_generated_rules = []
        num_failures = len(self.failed_cases_store)
        if num_failures < config.RULE_GEN_MIN_SUPPORT: return newly_generated_rules
        print(f"INFO: Attempting batch rule generation from {num_failures} stored failures.")

        embeddings = []; valid_indices = []
        for i, case in enumerate(self.failed_cases_store):
             q_info = case.get("question_info", {}); emb = q_info.get("embedding")
             if emb is not None and isinstance(emb, torch.Tensor) and emb.ndim == 2 and emb.shape[0] == 1:
                  embeddings.append(emb.numpy().flatten()); valid_indices.append(i)
             # else: print(f"Warning: Skipping failure case {i} due to missing/invalid embedding.")

        if len(valid_indices) < config.RULE_GEN_MIN_SUPPORT:
             print("INFO: Batch rule generation skipped: Not enough valid embeddings."); self.failed_cases_store.clear(); return newly_generated_rules
        embeddings_np = np.vstack(embeddings)

        try:
            clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=1.0 - config.RULE_GEN_CLUSTER_THRESHOLD, metric='cosine', linkage='average')
            labels = clustering.fit_predict(embeddings_np)
            num_clusters = (max(labels) + 1) if labels.size > 0 and max(labels) > -1 else 0
            print(f"INFO: Clustering resulted in {num_clusters} potential clusters.")
        except Exception as e: print(f"Error during clustering: {e}"); self.failed_cases_store.clear(); return newly_generated_rules

        generated_rules_in_batch = []
        for i in range(num_clusters):
            cluster_member_indices = [valid_indices[idx] for idx, label in enumerate(labels) if label == i]
            if len(cluster_member_indices) >= config.RULE_GEN_MIN_SUPPORT:
                # print(f"\nINFO: Analyzing Cluster {i} with {len(cluster_member_indices)} members.")
                cluster_cases = [self.failed_cases_store[idx] for idx in cluster_member_indices]
                common_query_pred = self._find_common_query_predicate(cluster_cases)
                if common_query_pred:
                    # print(f"  - Common query predicate: '{common_query_pred}'")
                    sample_facts = cluster_cases[0]['initial_facts']
                    generated_rule = self._apply_templates(common_query_pred, sample_facts, existing_rules + generated_rules_in_batch)
                    if generated_rule:
                         # print(f"  - Generated candidate Rule {generated_rule.id} for cluster {i}.")
                         newly_generated_rules.append(generated_rule); generated_rules_in_batch.append(generated_rule)
                    # else: print(f"  - No suitable template found or rule was redundant for cluster {i}.")
                # else: print(f"  - Could not determine a common query predicate for cluster {i}.")

        print(f"INFO: Finished batch processing. Generated {len(newly_generated_rules)} new rules.")
        self.failed_cases_store.clear()
        return newly_generated_rules

    def _find_common_query_predicate(self, cases: List[Dict]) -> Optional[str]:
        """Finds the most frequent query predicate among a list of failure cases."""
        query_preds = []
        for case in cases:
             q_info = case.get("question_info", {}); query = q_info.get("query")
             if query and isinstance(query, tuple) and len(query) == 3 and isinstance(query[0], str): query_preds.append(query[0])
        if not query_preds: return None
        predicate_counts = Counter(query_preds); most_common = predicate_counts.most_common(1)
        return most_common[0][0] if most_common else None

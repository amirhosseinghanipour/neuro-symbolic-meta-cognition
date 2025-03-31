from typing import Dict, Any

from neural_component import BertExtractor
from symbolic_component import SymbolicReasoner, Fact

class NeuralToSymbolicTranslator:
    """
    Translates the structured output from the neural component into symbolic facts
    for the symbolic reasoner.
    """

    def translate(self, neural_output: Dict[str, Any], reasoner: SymbolicReasoner) -> int:
        """
        Processes the dictionary from BertExtractor and adds corresponding Fact objects
        to the provided SymbolicReasoner instance.

        Iterates through the 'relations' list in the neural output.

        Args:
            neural_output (Dict[str, Any]): The dictionary returned by BertExtractor.
                                            Expected keys: 'relations' (list of tuples).
            reasoner (SymbolicReasoner): The reasoner instance to add facts to.

        Returns:
            int: The number of facts successfully added, or -1 on input format error.
        """
        added_facts_count = 0
        if not isinstance(neural_output, dict) or 'relations' not in neural_output:
             print("Error: Invalid neural_output format for translation.")
             return -1
        if not isinstance(reasoner, SymbolicReasoner):
             print("Error: Invalid reasoner object provided for translation.")
             return -1

        relations = neural_output.get('relations', [])
        if not isinstance(relations, list): relations = []

        for relation in relations:
            if isinstance(relation, tuple) and len(relation) == 3:
                predicate, subject, obj = relation
                if isinstance(predicate, str) and isinstance(subject, str):
                    try:
                        fact = Fact(predicate=predicate, subject=subject, object=obj)
                        if reasoner.add_fact(fact): added_facts_count += 1
                    except Exception as e: print(f"Error creating Fact from relation {relation}: {e}")
                # else: print(f"Warning: Skipping invalid relation format: {relation}")
            # else: print(f"Warning: Skipping invalid relation format: {relation}")

        return added_facts_count

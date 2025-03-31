import torch
from transformers import pipeline, AutoTokenizer, AutoModel
import re
from typing import List, Dict, Tuple, Any, Optional
from collections import defaultdict
from datetime import datetime, timedelta

from config import config
from utils import TimeUtils

class BertExtractor:
    """
    Extracts features, temporal entities, and relations from text.

    Uses a base BERT model for embeddings and a separate NER pipeline for entity recognition.
    Applies heuristics based on NER output and keywords to infer relations and identify queries.
    Intended for use within the NeuroSymbolicPipeline.
    """
    def __init__(self,
                 base_model_name: str = config.BASE_BERT_MODEL,
                 ner_model_name: str = config.NER_MODEL,
                 device: torch.device = config.DEVICE):
        """
        Initializes the BertExtractor with a base model and an NER pipeline.

        Args:
            base_model_name (str): Name of the pre-trained model for embeddings.
            ner_model_name (str): Name of the pre-trained model for NER pipeline.
            device (torch.device): The device to run models on.
        """
        print(f"Initializing BertExtractor...")
        print(f"  Loading base BERT model: {base_model_name}...")
        try:
            self.base_tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            self.base_model = AutoModel.from_pretrained(base_model_name)
            self.device = device
            pipeline_device_id = 0 if self.device.type == 'cuda' else -1
            self.base_model.to(self.device)
            self.base_model.eval()
            print("  Base BERT model loaded.")
        except Exception as e:
            print(f"ERROR: Failed to load base BERT model '{base_model_name}'. Error: {e}")
            raise e

        print(f"  Loading NER pipeline using model: {ner_model_name}...")
        try:
            self.ner_pipeline = pipeline("ner", model=ner_model_name, tokenizer=ner_model_name,
                                         aggregation_strategy="simple", device=pipeline_device_id)
            print("  NER pipeline loaded.")
        except Exception as e:
            print(f"ERROR: Failed to load NER pipeline '{ner_model_name}'. Error: {e}")
            raise e
        print("BertExtractor Initialized.")


    def get_embeddings(self, text: str) -> Optional[torch.Tensor]:
        """
        Generates the embedding for the [CLS] token using the base BERT model.

        Args:
            text (str): The input text string.

        Returns:
            Optional[torch.Tensor]: A tensor containing the [CLS] embedding (shape [1, hidden_size]),
                                    moved to CPU, or None if embedding fails.
        """
        try:
            inputs = self.base_tokenizer(
                text, return_tensors='pt', max_length=config.MAX_LENGTH,
                truncation=True, padding='max_length'
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.base_model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :]
            return cls_embedding.cpu()
        except Exception as e:
            print(f"Error generating embedding for text: '{text}'. Error: {e}")
            return None

    def _run_ner(self, text: str) -> List[Dict]:
        """
        Runs the NER pipeline and handles potential errors.

        Args:
            text (str): The input text.

        Returns:
            List[Dict]: A list of dictionaries representing identified entities,
                        or an empty list on error.
        """
        try:
            ner_results = self.ner_pipeline(text)
            if isinstance(ner_results, list) and all(isinstance(item, dict) for item in ner_results):
                 return ner_results
            else:
                 print(f"Warning: Unexpected NER pipeline output format: {type(ner_results)}")
                 return []
        except Exception as e:
            print(f"Error running NER pipeline on text: '{text}'. Error: {e}")
            return []

    def _parse_entities(self, ner_results: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Parses NER results into structured time and duration entities using TimeUtils.

        Args:
            ner_results (List[Dict]): The raw output from the NER pipeline.

        Returns:
            Dict[str, List[Dict]]: A dictionary containing lists of parsed 'times' and 'durations'.
                                   Each entity dict includes text, parsed value, span, and NER label.
        """
        parsed_entities_dict = {'times': [], 'durations': []}
        processed_spans = set()
        ner_results.sort(key=lambda x: x.get('start', 0))

        for entity in ner_results:
            entity_text = entity.get('word')
            entity_text = entity_text.replace("##", "").strip() if entity_text else None
            entity_label = entity.get('entity_group')
            start_char, end_char = entity.get('start'), entity.get('end')

            if not all([entity_text, entity_label, start_char is not None, end_char is not None]): continue
            span = (start_char, end_char)
            if any(p_start <= start_char and end_char <= p_end for p_start, p_end in processed_spans): continue

            parsed_value = None; entity_type = None
            parsed_time = TimeUtils.parse_time(entity_text)
            if parsed_time:
                 parsed_value = parsed_time; entity_type = 'times'
            else:
                 parsed_duration = TimeUtils.parse_duration(entity_text)
                 if parsed_duration:
                      parsed_value = parsed_duration; entity_type = 'durations'

            if entity_type and parsed_value:
                 entity_data = {
                     'text': entity_text, 'value': parsed_value,
                     'start': start_char, 'end': end_char, 'ner_label': entity_label
                 }
                 parsed_entities_dict[entity_type].append(entity_data)
                 processed_spans.add(span)

        return parsed_entities_dict


    def _assign_relations(self, question: str, entities: Dict[str, List[Dict]]) -> List[Tuple[str, str, Any]]:
        """
        Assigns relations (symbolic facts) based on extracted entities and keywords using heuristics.

        Args:
            question (str): The original question text.
            entities (Dict[str, List[Dict]]): Parsed time and duration entities.

        Returns:
            List[Tuple[str, str, Any]]: A list of relation tuples (predicate, subject, object).
        """
        relations = []
        event_name = "event1"
        question_lower = question.lower()
        times = entities['times']; durations = entities['durations']
        times.sort(key=lambda x: x['start']); durations.sort(key=lambda x: x['start'])

        start_time_assigned = False; end_time_assigned = False; event_time_assigned = False

        if len(times) == 1:
            t_ent = times[0]
            context_before = question[:t_ent['start']].lower()
            if "start" in context_before or "begins" in context_before:
                relations.append(('start_time', event_name, t_ent['value'])); start_time_assigned = True
            elif "depart" in context_before or "leaves at" in context_before:
                relations.append(('departure_time', event_name, t_ent['value']))
            elif "end" in context_before or "finish" in context_before or "until" in context_before:
                 relations.append(('end_time', event_name, t_ent['value'])); end_time_assigned = True
            elif "meeting at" in context_before or "flight is at" in context_before or "deadline is" in context_before:
                 relations.append(('event_time', event_name, t_ent['value'])); event_time_assigned = True
            else:
                 relations.append(('event_time', event_name, t_ent['value'])); event_time_assigned = True
        elif len(times) >= 2:
             t_ent1, t_ent2 = times[0], times[1]
             context_before_t1 = question[:t_ent1['start']].lower()
             context_between = question[t_ent1['end']:t_ent2['start']].lower()
             if "from" in context_before_t1 and ("to" in context_between or "until" in context_between):
                  relations.append(('start_time', event_name, t_ent1['value'])); start_time_assigned = True
                  relations.append(('end_time', event_name, t_ent2['value'])); end_time_assigned = True
             else:
                  relations.append(('start_time', event_name, t_ent1['value'])); start_time_assigned = True
                  relations.append(('end_time', event_name, t_ent2['value'])); end_time_assigned = True

        if durations:
             d_ent = durations[0]
             window = 25
             context_around = question[max(0, d_ent['start']-window) : min(len(question), d_ent['end']+window)].lower()
             travel_kws = ["get there", "travel", "drive", "walk", "commute", "journey", "trip"]
             event_dur_kws = ["runs for", "lasts", "duration of", "takes", "need", "review"]

             if any(kw in context_around for kw in travel_kws):
                  relations.append(('travel_time', event_name, d_ent['value']))
             elif any(kw in context_around for kw in event_dur_kws):
                  if start_time_assigned or event_time_assigned:
                       relations.append(('duration', event_name, d_ent['value']))
                  elif any(r[0] == 'departure_time' for r in relations):
                       relations.append(('travel_time', event_name, d_ent['value']))
                  else:
                       relations.append(('duration', event_name, d_ent['value']))
             elif not any(r[0] in ['travel_time', 'duration'] for r in relations):
                  if start_time_assigned or event_time_assigned:
                       relations.append(('duration', event_name, d_ent['value']))
                  else:
                       relations.append(('duration', event_name, d_ent['value']))

        return relations

    def _identify_query(self, question: str, relations: List[Tuple[str, str, Any]]) -> Optional[Tuple[str, str, str]]:
        """
        Identifies the query (what the question is asking for) based on question patterns.

        Args:
            question (str): The original question text.
            relations (List[Tuple[str, str, Any]]): The list of assigned relations (currently unused here).

        Returns:
            Optional[Tuple[str, str, str]]: The query tuple (predicate, subject, '?') or None.
        """
        question_lower = question.lower()
        event_name = "event1"

        if re.search(r'when should (i|we) leave|latest departure time', question_lower):
            return ('departure_time', event_name, '?')
        if re.search(r'when should (i|we) start.*review|when to start.*review', question_lower):
             return ('start_work_time', event_name, '?')
        if re.search(r'when should (i|we) start', question_lower):
             return ('start_time', event_name, '?')
        if re.search(r'when will (i|we) arrive|what time do (i|we) arrive', question_lower):
             return ('arrival_time', event_name, '?')
        if re.search(r'what time does it finish|when does it end', question_lower):
             return ('end_time', event_name, '?')
        if re.search(r'how long|duration of', question_lower):
             return ('duration', event_name, '?')
        if "departure time" in question_lower: return ('departure_time', event_name, '?')
        if "start work" in question_lower: return ('start_work_time', event_name, '?')
        if "arrival time" in question_lower: return ('arrival_time', event_name, '?')
        if "end time" in question_lower: return ('end_time', event_name, '?')

        print(f"Warning: Could not identify query for question: '{question}'")
        return None


    def extract_entities_relations(self, question: str) -> Dict[str, Any]:
        """
        Performs entity/relation extraction using NER pipeline and revised heuristics.

        Args:
            question (str): The input natural language question string.

        Returns:
            Dict[str, Any]: A dictionary containing:
                            'question', 'entities' (parsed times/durations),
                            'relations' (heuristic tuples), 'query' (identified tuple or None),
                            'embedding' (sentence embedding tensor or None).
        """
        ner_results = self._run_ner(question)
        entities = self._parse_entities(ner_results)
        relations = self._assign_relations(question, entities)
        query_variable = self._identify_query(question, relations)
        embedding = self.get_embeddings(question)

        return {
            "question": question,
            "entities": entities,
            "relations": relations,
            "query": query_variable,
            "embedding": embedding
        }

    def fine_tune_step(self, batch_questions: List[str], batch_labels: List[Any], optimizer, criterion):
        """Placeholder structure for fine-tuning in future"""
        print("Warning: BertExtractor.fine_tune_step is a placeholder and not implemented for actual training.")
        pass

import os
from typing import Optional
import dateparser
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification, AutoModelForSequenceClassification

# Lazy-loaded globals
intent_pipe: Optional[object] = None
ner_pipe: Optional[object] = None


def _local_model_path(name: str) -> Optional[str]:
    """Return local model path if present in repo root or nearby directories."""
    # Prioritize repo root (one level up from api/)
    here = os.path.dirname(__file__)
    repo_root = os.path.abspath(os.path.join(here, ".."))
    
    candidates = [
        os.path.join(repo_root, name),  # <-- Check repo root first
        os.path.join(here, name),
        os.path.join(os.getcwd(), name),
    ]
    for p in candidates:
        p_abs = os.path.abspath(p)
        if os.path.isdir(p_abs):
            print(f"Found {name} at: {p_abs}")
            return p_abs
    print(f"Model {name} not found in: {candidates}")
    return None


def _ensure_models_loaded():
    global intent_pipe, ner_pipe
    if intent_pipe is not None and ner_pipe is not None:
        return

    # Intent model
    intent_path = _local_model_path("model_intent")
    if intent_path:
        intent_tokenizer = AutoTokenizer.from_pretrained(intent_path)
        intent_model = AutoModelForSequenceClassification.from_pretrained(intent_path)
        intent_pipe = pipeline("text-classification", model=intent_model, tokenizer=intent_tokenizer)
    else:
        # allow explicit HF repo via env var, otherwise fail with clear message
        hf_intent = os.environ.get("HF_MODEL_INTENT")
        if hf_intent:
            intent_tokenizer = AutoTokenizer.from_pretrained(hf_intent)
            intent_model = AutoModelForSequenceClassification.from_pretrained(hf_intent)
            intent_pipe = pipeline("text-classification", model=intent_model, tokenizer=intent_tokenizer)
        else:
            raise RuntimeError("Intent model not found locally. Place a trained model in 'model_intent' or set HF_MODEL_INTENT env var.")

    # NER model
    ner_path = _local_model_path("model_ner")
    if ner_path:
        ner_tokenizer = AutoTokenizer.from_pretrained(ner_path)
        ner_model = AutoModelForTokenClassification.from_pretrained(ner_path)
        ner_pipe = pipeline("token-classification", model=ner_model, tokenizer=ner_tokenizer, aggregation_strategy="max")
    else:
        hf_ner = os.environ.get("HF_MODEL_NER")
        if hf_ner:
            ner_tokenizer = AutoTokenizer.from_pretrained(hf_ner)
            ner_model = AutoModelForTokenClassification.from_pretrained(hf_ner)
            ner_pipe = pipeline("token-classification", model=ner_model, tokenizer=ner_tokenizer, aggregation_strategy="max")
        else:
            raise RuntimeError("NER model not found locally. Place a trained model in 'model_ner' or set HF_MODEL_NER env var.")


def analyze(text: str):
    _ensure_models_loaded()
    
    # Split by "and" to handle multiple tasks
    # e.g., "call mom and get milk" -> ["call mom", "get milk"]
    import re
    parts = re.split(r'\s+and\s+', text, flags=re.IGNORECASE)
    
    results = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        
        # Run intent classification on this part
        intent = intent_pipe(part)[0]["label"]
        
        # Run NER on this part
        entities = ner_pipe(part)
        
        # Build result for this task
        result = {"intent": intent}
        for ent in entities:
            # entity_group is like 'TASK' or 'DEADLINE' — normalize to lowercase keys
            key = ent.get("entity_group") or ent.get("entity") or "entity"
            key = key.lower()
            result.setdefault(key, "")
            result[key] += ent.get("word", "") + " "
        
        # Clean up trailing spaces
        for key in result:
            if isinstance(result[key], str):
                result[key] = result[key].strip()
        
        # Post-process to separate deadline and priority if mixed
        # If "this is" is mistakenly added to the end of a deadline, remove it.
        if "deadline" in result and "priority" in result and result["deadline"].endswith(" this is"):
            result["deadline"] = result["deadline"].removesuffix(" this is").strip()

        # Post-process deadline entities to parse dates
        if "deadline" in result and isinstance(result["deadline"], str):
            # Settings to prefer future dates and be strict about what is parsed
            settings = {
                'PREFER_DATES_FROM': 'future'
            }
            parsed_date = dateparser.parse(result["deadline"], settings=settings)
            if parsed_date:
                result["deadline"] = parsed_date.strftime('%Y-%m-%dT%H:%M:%S')
        
        results.append(result)
    
    # Return list if multiple tasks, single dict if one
    if len(results) == 1:
        return results[0]
    return results

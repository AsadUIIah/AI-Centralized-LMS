import os
import random

import spacy
import torch
from transformers import T5ForConditionalGeneration, T5TokenizerFast

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Adjust this path to wherever you unzipped the model from Colab.
MODEL_DIR = os.path.join(os.path.dirname(__file__), "ml_models", "t5-qg-finetuned")

MAX_INPUT_LEN = 384
MAX_TARGET_LEN = 64

# Entity types worth turning into quiz answers. Feel free to tweak based on
# what your course material looks like.
RELEVANT_ENTITY_LABELS = {
    "PERSON", "ORG", "GPE", "LOC", "DATE", "EVENT", "WORK_OF_ART",
    "LAW", "NORP", "PRODUCT", "FAC",
}

# ---------------------------------------------------------------------------
# Lazy-loaded globals (populated by load_models(), called from apps.py)
# ---------------------------------------------------------------------------

_nlp = None
_tokenizer = None
_model = None
_device = None


def load_models():
    """Load spaCy + the fine-tuned T5 model into memory. Call once at Django startup."""
    global _nlp, _tokenizer, _model, _device

    if _nlp is not None:
        return  # already loaded

    _nlp = spacy.load("en_core_web_sm")

    _tokenizer = T5TokenizerFast.from_pretrained(MODEL_DIR)
    _model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR)

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    _model = _model.to(_device)
    _model.eval()

    print(f"[ml_utils] Loaded fine-tuned QG model from {MODEL_DIR} on {_device}")


def _ensure_loaded():
    if _model is None:
        raise RuntimeError(
            "ML models not loaded yet. Make sure load_models() is called from "
            "quiz/apps.py's ready() method."
        )


# ---------------------------------------------------------------------------
# Step 1: Answer extraction
# ---------------------------------------------------------------------------

def extract_candidate_answers(text, max_candidates=20):
    """
    Pull candidate answers out of course material text.

    Returns a list of dicts: {"text": str, "label": str}
    label is a spaCy entity type (e.g. "PERSON", "DATE") or "NOUN_CHUNK" as a fallback.
    """
    _ensure_loaded()
    doc = _nlp(text)

    candidates = []
    seen = set()

    # Prefer named entities - they make more specific, better quiz answers.
    for ent in doc.ents:
        if ent.label_ in RELEVANT_ENTITY_LABELS:
            key = ent.text.strip().lower()
            if key and key not in seen:
                seen.add(key)
                candidates.append({"text": ent.text.strip(), "label": ent.label_})

    # Fall back to noun chunks if we don't have enough named entities
    # (common with more conceptual/technical course material that has few named entities).
    if len(candidates) < max_candidates:
        for chunk in doc.noun_chunks:
            key = chunk.text.strip().lower()
            # skip pronouns, very short/very long chunks
            if key and key not in seen and 2 <= len(chunk.text) <= 60:
                seen.add(key)
                candidates.append({"text": chunk.text.strip(), "label": "NOUN_CHUNK"})
            if len(candidates) >= max_candidates:
                break

    return candidates[:max_candidates]


# ---------------------------------------------------------------------------
# Step 2: Question generation
# ---------------------------------------------------------------------------

def generate_question(answer, context, max_length=64, num_beams=4):
    """Generate a question for a given (answer, context) pair using the fine-tuned model."""
    _ensure_loaded()
    input_text = f"answer: {answer} context: {context}"
    input_ids = _tokenizer(
        input_text, return_tensors="pt", truncation=True, max_length=MAX_INPUT_LEN
    ).input_ids.to(_device)

    with torch.no_grad():
        output_ids = _model.generate(input_ids, max_length=max_length, num_beams=num_beams)

    return _tokenizer.decode(output_ids[0], skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Step 3: Distractor generation (wrong options for MCQs)
# ---------------------------------------------------------------------------

def generate_distractors(answer, all_candidates, num_distractors=3):
    """
    Pick plausible wrong answers for a multiple-choice question.

    Strategy: prefer other candidates of the *same* entity type as the correct answer
    (e.g. other PERSON names as distractors for a PERSON-type answer), since same-type
    wrong options tend to be more plausible than random ones. Falls back to any other
    candidate if there aren't enough same-type options.
    """
    answer_label = None
    for c in all_candidates:
        if c["text"] == answer:
            answer_label = c["label"]
            break

    same_type = [c["text"] for c in all_candidates
                 if c["text"] != answer and c["label"] == answer_label]
    other_type = [c["text"] for c in all_candidates
                  if c["text"] != answer and c["label"] != answer_label]

    random.shuffle(same_type)
    random.shuffle(other_type)

    distractors = (same_type + other_type)[:num_distractors]

    # Pad with generic placeholders if the material was too short to have enough candidates.
    while len(distractors) < num_distractors:
        distractors.append(f"None of the above ({len(distractors) + 1})")

    return distractors


# ---------------------------------------------------------------------------
# Main entry point: ties everything together
# ---------------------------------------------------------------------------

def generate_quiz_questions(material, num_questions=5):
    """
    Main function called from views.py.

    Returns a list of dicts:
    [{"question": str, "options": [str, str, str, str], "answer": str}, ...]
    """
    _ensure_loaded()

    candidates = extract_candidate_answers(material, max_candidates=max(num_questions * 3, 10))

    if not candidates:
        return []

    # Pick up to num_questions distinct answers to build questions around.
    random.shuffle(candidates)
    chosen = candidates[:num_questions]

    results = []
    for candidate in chosen:
        answer_text = candidate["text"]
        question_text = generate_question(answer_text, material)
        distractors = generate_distractors(answer_text, candidates, num_distractors=3)

        options = distractors + [answer_text]
        random.shuffle(options)

        results.append({
            "question": question_text,
            "options": options,
            "answer": answer_text,
        })

    return results

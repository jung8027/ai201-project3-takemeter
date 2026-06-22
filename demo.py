"""
TakeMeter — Demo Script
AI201 Project 3

Run this as a new cell in your Colab notebook AFTER Section 3 (fine-tuning) is complete.
The fine-tuned model must be saved at ./takemeter-model/ (this happens automatically at the
end of Section 3).

RECORDING GUIDE
---------------
This script is structured as five sections. Read the NARRATOR lines aloud while running
each section. Total target: 3-5 minutes.

  Section 1 (~30s)  — Setup and intro
  Section 2 (~60s)  — Classify 5 posts, show labels and confidence
  Section 3 (~45s)  — Narrate the correct prediction
  Section 4 (~60s)  — Narrate the incorrect prediction
  Section 5 (~60s)  — Walk through the evaluation report numbers

Run one section at a time so viewers can see each output.
"""

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — Load the fine-tuned model
# ─────────────────────────────────────────────────────────────────────────────
#
# NARRATOR:
#   "This is TakeMeter, a fine-tuned DistilBERT classifier that labels Reddit posts
#    from r/aiengineering as either Discussion_General or Project_Development.
#    Let me load the fine-tuned model and run a few posts through it."
#

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_PATH = "./takemeter-model/"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

ID_TO_LABEL = {0: "Discussion_General", 1: "Project_Development"}

def classify(text):
    """Return (predicted_label, confidence) for a post title."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.nn.functional.softmax(logits, dim=-1)[0]
    pred_id = probs.argmax().item()
    return ID_TO_LABEL[pred_id], round(probs[pred_id].item(), 2)

print("✅ Model loaded from", MODEL_PATH)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — Classify 5 posts
# ─────────────────────────────────────────────────────────────────────────────
#
# NARRATOR:
#   "I'll run five posts through the classifier — posts that were in the test set.
#    The first two are Discussion_General posts. The last three are Project_Development
#    posts. Watch what the model predicts and the confidence score for each."
#

demo_posts = [
    {
        "text":       "Is your team reviewing AI-generated code?",
        "true_label": "Discussion_General",
    },
    {
        "text":       "Are clients starting to underestimate software complexity because of AI?",
        "true_label": "Discussion_General",
    },
    {
        "text":       "Has anyone found a reliable way to enforce strict JSON outputs at scale?",
        "true_label": "Project_Development",
    },
    {
        "text":       "AI image detection models?",
        "true_label": "Project_Development",
    },
    {
        "text":       "Working in AI for 8 Years Taught Me Why Both AI Doomers and Optimists Miss the Point.",
        "true_label": "Project_Development",
    },
]

print("=" * 72)
print(f"{'POST':<52} {'PREDICTED':<22} {'CONF':>4}  {'OK?'}")
print("=" * 72)
for p in demo_posts:
    pred, conf = classify(p["text"])
    correct = "✓" if pred == p["true_label"] else "✗"
    short = (p["text"][:49] + "…") if len(p["text"]) > 50 else p["text"]
    print(f"{short:<52} {pred:<22} {conf:>4}  {correct}")
print("=" * 72)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — Narrate the correct prediction
# ─────────────────────────────────────────────────────────────────────────────
#
# NARRATOR:
#   "Let me zoom in on the first post: 'Is your team reviewing AI-generated code?'
#
#    The model correctly predicted Discussion_General with 0.92 confidence.
#    Why does this work? The title is a general workplace question — it asks whether
#    teams are doing something, with no specific system or artifact being built.
#    Vocabulary like 'your team' and 'reviewing' matches the Discussion_General
#    pattern the model saw in 166 training examples of this class.
#    There are no tool names, no architecture terms, nothing that signals someone
#    is actively building. The model recognized this pattern cleanly."
#

post_correct = "Is your team reviewing AI-generated code?"
pred, conf = classify(post_correct)
print("\n── Correct prediction ──────────────────────────────────────────────────")
print(f"  Post:       \"{post_correct}\"")
print(f"  True label: Discussion_General")
print(f"  Predicted:  {pred}  (confidence: {conf})")
print(f"  Result:     {'✓ Correct' if pred == 'Discussion_General' else '✗ Wrong'}")
print()
print("  Why it works: general workplace question, no artifact named,")
print("  no project-specific vocabulary → clear Discussion_General signal.")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — Narrate the incorrect prediction
# ─────────────────────────────────────────────────────────────────────────────
#
# NARRATOR:
#   "Now let's look at a failure. 'Has anyone found a reliable way to enforce
#    strict JSON outputs at scale?'
#
#    The true label is Project_Development — this person is building a production
#    LLM pipeline and needs structured outputs. But the model predicted
#    Discussion_General with 0.60 confidence.
#
#    Why did it fail? The phrasing 'Has anyone found' is an open question pattern
#    that looks exactly like a Discussion_General post. The model latched onto that
#    framing and missed the technical specificity underneath it — enforcing JSON
#    at scale is a concrete engineering problem, not a general discussion.
#
#    At 0.60 confidence, the model was nearly uncertain. The imbalance in the
#    training data — 166 Discussion_General examples versus only 27 Project_Development —
#    tipped the decision toward the majority class whenever the model wasn't sure.
#    This is the core failure mode: when in doubt, predict Discussion_General."
#

post_wrong = "Has anyone found a reliable way to enforce strict JSON outputs at scale?"
pred, conf = classify(post_wrong)
print("\n── Incorrect prediction ────────────────────────────────────────────────")
print(f"  Post:       \"{post_wrong}\"")
print(f"  True label: Project_Development")
print(f"  Predicted:  {pred}  (confidence: {conf})")
print(f"  Result:     {'✓ Correct' if pred == 'Project_Development' else '✗ Wrong'}")
print()
print("  Why it failed: interrogative phrasing ('Has anyone found') matches")
print("  Discussion_General patterns. Model had low confidence (0.60) and")
print("  defaulted to the majority class due to 86%/14% training imbalance.")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — Evaluation report walkthrough
# ─────────────────────────────────────────────────────────────────────────────
#
# NARRATOR:
#   "Here's a summary of the full evaluation on the 42-example test set.
#
#    The fine-tuned model reached 85.7% overall accuracy — 23.8 percentage points
#    above the zero-shot Groq baseline at 61.9%.
#
#    But overall accuracy is misleading here. Looking at the per-class breakdown,
#    the model achieved 100% recall on Discussion_General — it got every single
#    one right. For Project_Development, recall was 0%. Zero out of six.
#    The model never predicted Project_Development once.
#
#    The baseline actually did better on the minority class: it found 1 of 6
#    Project_Development posts (17% recall, F1 of 0.11) while the fine-tuned
#    model found none.
#
#    The confusion matrix tells the whole story. Every Project_Development post
#    landed in the Discussion_General column. The fine-tuned model learned one
#    thing: predict Discussion_General. That's not a classifier — that's a
#    majority-class predictor. The root cause is the 86%/14% label split, which
#    gave the model only 27 Project_Development training examples, several of
#    which were mislabeled due to inconsistent Reddit flair assignments."
#

import json, pathlib

results = json.loads(
    pathlib.Path("documents/evaluation_results.json").read_text()
)

print("\n── Evaluation Report ───────────────────────────────────────────────────")
print(f"  Test set size      : {results['test_set_size']} examples")
print(f"  Baseline accuracy  : {results['baseline_accuracy']:.1%}")
print(f"  Fine-tuned accuracy: {results['finetuned_accuracy']:.1%}")
print(f"  Improvement        : +{results['improvement']:.1%}")
print()
print("  Per-class (fine-tuned):")
print("    Discussion_General  — precision 0.86  recall 1.00  F1 0.92")
print("    Project_Development — precision 0.00  recall 0.00  F1 0.00")
print()
print("  Per-class (baseline):")
print("    Discussion_General  — precision 0.83  recall 0.69  F1 0.76")
print("    Project_Development — precision 0.08  recall 0.17  F1 0.11")
print()
print("  Confusion matrix (fine-tuned):")
print("                        Pred DG   Pred PD")
print("    True DG                 36         0")
print("    True PD                  6         0   ← all wrong")
print()
print("  Conclusion: model predicts Discussion_General for every post.")
print("  Overall accuracy gain is real; minority-class performance is zero.")

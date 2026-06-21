# TakeMeter — Planning Document
## AI201 · Project 3

---

## Community Choice

**Community:** r/aiengineering (Reddit)

I chose r/aiengineering because it contains a natural split between two types of posts: people sharing what they're building vs. people asking general questions or discussing ideas. This makes it a good candidate for a classifier that distinguishes output-oriented content from input-oriented content. The community is active enough on public Reddit to collect 200+ labeled examples from post titles alone.

---

## Label Taxonomy

I will classify each post into one of two categories:

### `Discussion_General`
Posts that discuss AI engineering topics in a general, conceptual, career-focused, or question-oriented way — not tied to a specific project the author is actively building. Includes questions, opinion threads, news commentary, job/career posts, and general tool comparisons.

**Examples:**
- "Are clients starting to underestimate software engineers when discussing AI Engineering solutions?" → opinion question about industry dynamics, no specific project
- "What's the best resource to learn LangChain in 2024?" → seeking general information, not building anything specific

### `Project_Development`
Posts where the author is actively developing, building, deploying, or demonstrating a specific AI engineering artifact — a tool, pipeline, model, agent, API, or application. Also includes posts seeking help for a specific technical implementation.

**Examples:**
- "Multi-tenant AI Customer Support Agent (with ticketing integration)" → describes a specific system being built
- "I built a RAG pipeline that handles 10k documents — here's how" → a specific project with technical implementation

---

## Hard Edge Cases

**The hardest anticipated boundary:** posts where the author asks a question *about* building something vs. posts where the author *is* building something.

**Example ambiguous post:** *"What's the best way to structure a multi-agent pipeline for document processing?"*

This could be:
- `Discussion_General` — asking a general architectural question, no evidence of an active project
- `Project_Development` — seeking implementation help for a specific system they're building

**Decision rule:** If the post title names a specific artifact the author is creating (a named tool, a specific integration, a described system) — label `Project_Development`, even if phrased as a question. If the title asks about a concept, approach, or comparison without naming something the author is actively building — label `Discussion_General`. Framing alone ("how do I" vs. "I built") is not sufficient; the key signal is whether a specific artifact is referenced.

**Secondary edge case:** short or ambiguous titles like "Career", "Fine tuning learning ai", or truncated text. Decision rule: if the original flair maps to a category and there's no contradictory signal in the title, use the flair-derived label. Flag these as low-confidence labels.

---

## Data Collection Plan

**Source:** Reddit posts from r/aiengineering collected via the public Reddit API. I will use post titles as the primary text feature — titles are short, self-contained, and carry the post's intent without requiring the body.

**Volume target:** 200+ labeled examples, aiming for at least 30% minority class (Project_Development).

**Labeling approach:** Use Reddit post flairs as proxy labels, then consolidate flair variants into the two categories via a remapping table. The subreddit uses flairs like "Discussion", "Engineering", "RAGDiscussion", "Data", etc. I will create a full mapping from each observed flair to one of the two labels. Posts with unmapped or missing flairs will default to `Discussion_General` unless manually reviewed.

**Expected challenges:**
- Short titles (1–5 words) that lack enough signal
- Posts that blend discussion and project content ("I'm building X — is this approach good?")
- Flair inconsistency across users

---

## Evaluation Metrics

**Primary metric:** Per-class F1 score for both classes, not overall accuracy alone.

**Why accuracy is insufficient:** With a 74%/26% class split, a model that predicts `Discussion_General` for every post would achieve 74% accuracy while being completely useless. Overall accuracy rewards the majority-class heuristic and hides minority-class failure.

**What to use instead:**
- **Recall for `Project_Development`:** The harder, underrepresented class. High recall means the model is actually finding these posts rather than ignoring them. Target ≥ 0.60.
- **F1 for `Project_Development`:** Harmonic mean of precision and recall — the single most useful number for the minority class.
- **Macro-average F1:** Treats both classes equally regardless of support. This is the honest summary metric for an imbalanced task.
- **Confusion matrix:** Shows whether errors are directional (all PD misclassified as DG, or symmetric). A directional pattern tells you exactly which boundary the model hasn't learned.

**Why these are right for this task:** The goal is to distinguish two different types of posts. Systematically missing all `Project_Development` posts is a qualitative failure, not just a quantitative one — it means the classifier is not doing its job, even if overall accuracy looks acceptable.

---

## Definition of Success

A classifier is **genuinely useful** if it can surface `Project_Development` posts with enough reliability to be used as a first-pass filter:
- `Project_Development` recall ≥ 0.60 (catches at least 3 out of every 5 project posts)
- `Project_Development` F1 ≥ 0.50
- Overall accuracy ≥ 0.70

**Acceptable for the fine-tuned model:** matching or exceeding the baseline on `Project_Development` F1, with overall accuracy improvement of at least 3 percentage points.

**Not acceptable:** A model that predicts only `Discussion_General` — this would be useless regardless of accuracy score.

---

## Fine-Tuning Approach

**Base model:** `distilbert-base-uncased` — a small, fast transformer (66M parameters) that fine-tunes well on classification tasks with 100–500 examples.

**Training setup:**
- Split: 70% train / 15% validation / 15% test, stratified by label
- Epochs: 3
- Learning rate: 2e-5
- Batch size: 16
- Weight decay: 0.01
- Warmup steps: 50
- Tokenizer max length: 256

**Hyperparameter decision:** `num_train_epochs=3` is the planned choice. With ~140 training examples, more epochs risk overfitting to the training distribution. Fewer epochs risk underfitting given class imbalance.

---

## Baseline Plan

I will use Groq's `llama-3.3-70b-versatile` as a zero-shot classifier. The prompt will name the community, define each label, give one example per label, and instruct the model to output only the label name. Results will be collected by running the classifier on the same locked test set used for fine-tuning, then computing accuracy and per-class metrics.

---

## AI Tool Plan

**Label stress-testing:** Before annotating 200 examples, give Claude the two label definitions and edge case description and ask it to generate 8–10 post titles that sit at the boundary between `Discussion_General` and `Project_Development`. If any generated titles cannot be cleanly assigned using the decision rules above, tighten the definitions before starting annotation. Specifically target: short titles, question-phrased project posts, and career-adjacent technical questions.

**Annotation assistance:** Use Claude to propose an initial flair-to-label mapping from the full list of observed Reddit flair values — provide the label definitions and ask it to suggest a category for each flair. Review every proposed mapping against the actual posts in that flair before accepting it. Track which mappings were AI-suggested vs. manually verified (note in the notebook's `label_remapper` dict). Disclose this in the AI usage section of the README.

**Failure analysis:** After collecting wrong predictions from the fine-tuned model, paste all misclassified examples into Claude and ask it to identify common themes: similar post length, ambiguous vocabulary, a specific label pair that keeps being confused, short/low-information posts. Then verify the suggested patterns by re-reading the examples manually. Include what the AI found and whether I agreed or corrected it in the evaluation report.

---

*Document written before data collection. Updated before stretch features.*

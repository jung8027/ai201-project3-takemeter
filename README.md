# TakeMeter
### AI201 · Project 3

TakeMeter is a fine-tuned text classifier that categorizes posts from r/aiengineering into one of two labels: **Discussion_General** or **Project_Development**. It is built on `distilbert-base-uncased`, fine-tuned on a hand-remapped dataset of 276 Reddit post titles, and compared against a zero-shot Groq baseline.

---

## Community Choice

**Community:** r/aiengineering (Reddit)

I chose r/aiengineering because it contains a natural and meaningful split between two types of posts: people sharing what they are actively building and people asking general questions, sharing opinions, or discussing career topics. This makes the classification task non-trivial — many posts look similar on the surface — but grounded in a real distinction that would be useful for community health metrics or content recommendation. The community is large and active enough to collect labeled examples from public post titles without scraping private content.

---

## Label Taxonomy

| Label | Definition |
|---|---|
| `Discussion_General` | Posts that discuss AI engineering topics in a general, conceptual, career-focused, or question-oriented way — not tied to a specific project the author is actively building. Includes opinion threads, news commentary, career questions, and general tool comparisons. |
| `Project_Development` | Posts where the author is building, deploying, demonstrating, or seeking technical help for a specific AI engineering artifact — a tool, pipeline, model, agent, API, or application. |

### 2 Examples per Label

**`Discussion_General`**
1. *"Are clients starting to underestimate software complexity because of AI?"* — an opinion question about industry dynamics with no specific project or implementation attached
2. *"What's the difference between an AI engineer and an ML engineer?"* — a general conceptual/career question seeking information, not building anything

**`Project_Development`**
1. *"Multi-tenant AI Customer Support Agent (with ticketing integration)"* — a post describing a specific system being built with specific architectural features
2. *"I built SemanticCache a high-performance semantic caching library for Go"* — a post sharing a concrete engineering implementation

---

## Data Collection

**Source:** Posts from r/aiengineering scraped via the public Reddit API. The dataset (`documents/aiengineering_reddit_posts.csv`) contains post titles (column: `title`), scores, flair (column: `flair`), domain, submission time, author, user tags, and comment count. Only the post title was used as input text for classification.

**Labeling process:** Reddit post flairs served as proxy labels, then consolidated into the two categories via a remapping table defined in the notebook. `Discussion`, `Other`, `Highlight`, `Media`, `Hardware`, `Humor`, `Announcement`, and `General` all map to `Discussion_General`; `Engineering` and `Data` map to `Project_Development`. Posts with missing flairs defaulted to `Discussion_General`.

**Label distribution:**

| Label | Count | Proportion |
|---|---|---|
| `Discussion_General` | 237 | 85.9% |
| `Project_Development` | 39 | 14.1% |
| **Total** | **276** | — |

The dataset is heavily class-imbalanced: Discussion_General posts are 6.1× more common than Project_Development posts. All available public posts from the subreddit were collected; the imbalance reflects the community's natural flair distribution, not a collection shortcut. This imbalance directly caused the model to learn a majority-class heuristic (see Evaluation Report).

**Three difficult-to-label examples:**

1. *"Could u help me become an AI engineer?"* — **Labeled: Project_Development** (flair: `Engineering`). The phrasing "help me become" reads as a Discussion_General career question with no artifact named, but the Engineering flair maps to Project_Development. Kept the flair-derived label; likely annotation noise — the content is functionally a career post.

2. *"Any existing solutions to generate SVG icons at scale?"* — **Labeled: Project_Development** (flair: `Engineering`). "Any existing solutions" reads as a tool-recommendation request (Discussion_General), but the Engineering flair and project-specific goal suggest the author is building something concrete. Kept the flair-derived label because the scale requirement implies an active project need.

3. *"looking for a small model for multi-language text classification"* — **Labeled: Discussion_General** (flair: `Discussion`). Describes a specific technical requirement that implies an active project (Project_Development), but the Discussion flair and absence of a named artifact map it to Discussion_General. Kept the flair-derived label; borderline without the post body.

---

## Fine-Tuning Approach

**Base model:** `distilbert-base-uncased` — a lightweight transformer (66M parameters) that fine-tunes effectively on small classification datasets.

**Training platform:** Google Colab

**Training setup:**

| Hyperparameter | Value |
|---|---|
| num_train_epochs | 3 |
| learning_rate | 2e-5 |
| per_device_train_batch_size | 16 |
| per_device_eval_batch_size | 32 |
| weight_decay | 0.01 |
| warmup_steps | 50 |
| max_length (tokenizer) | 256 |
| eval_strategy | epoch |
| load_best_model_at_end | true |

**Train / val / test split:** 193 / 41 / 42 (70% / 15% / 15%), stratified by label. Training set: 166 Discussion_General, 27 Project_Development.

**Hyperparameter decision:** `num_train_epochs=3` was chosen over larger values. With only 27 Project_Development examples in training, additional epochs risk memorizing those examples rather than learning generalizable patterns. The learning rate of `2e-5` is the standard BERT fine-tuning starting point — it converges stably over 3 epochs on datasets of this size without overshooting.

---

## Baseline Description

**Model:** Groq `llama-3.3-70b-versatile`, zero-shot

**Prompt used:**

```
You are classifying posts from r/aiengineering.
Assign each post to exactly one of the following categories.

Discussion_General: Posts that are general discussions, questions, or news related to AI engineering that don't directly involve a specific project or development.
Example: "Are clients starting to underestimate software engineers when discussing AI Engineering solutions?"

Project_Development: Posts related to the actual development, implementation, or technical aspects of AI engineering projects, tools, or models.
Example: "Any existing solutions to generate SVG icons at scale?"

Respond with ONLY the label name.
Do not explain your reasoning.

Valid labels:
Discussion_General
Project_Development
```

**Collection:** The prompt was run on all 42 test set examples using `temperature=0` and `max_tokens=20`. Outputs were parsed by exact string match, then case-insensitive substring match as fallback. All 42 responses were parseable.

---

## Evaluation Report

### Overall Accuracy

| Model | Accuracy | Test Set Size |
|---|---|---|
| Zero-shot baseline (Groq llama-3.3-70b) | 61.9% | 42 |
| Fine-tuned DistilBERT | **85.7%** | 42 |
| Improvement | **+23.8 pp** | — |

### Per-Class Metrics — Fine-Tuned Model

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| `Discussion_General` | 0.86 | 1.00 | 0.92 | 36 |
| `Project_Development` | 0.00 | 0.00 | 0.00 | 6 |
| **Accuracy** | | | **0.86** | **42** |
| Macro avg | 0.43 | 0.50 | 0.46 | 42 |
| Weighted avg | 0.73 | 0.86 | 0.79 | 42 |

### Per-Class Metrics — Baseline

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| `Discussion_General` | 0.83 | 0.69 | 0.76 | 36 |
| `Project_Development` | 0.08 | 0.17 | 0.11 | 6 |
| **Accuracy** | | | **0.62** | **42** |
| Macro avg | 0.46 | 0.43 | 0.43 | 42 |
| Weighted avg | 0.73 | 0.62 | 0.67 | 42 |

### Confusion Matrix — Fine-Tuned Model

|  | Pred: Discussion_General | Pred: Project_Development |
|---|---|---|
| **True: Discussion_General** | 36 | 0 |
| **True: Project_Development** | 6 | 0 |

### Confusion Matrix — Baseline

|  | Pred: Discussion_General | Pred: Project_Development |
|---|---|---|
| **True: Discussion_General** | 25 | 11 |
| **True: Project_Development** | 5 | 1 |

![Confusion Matrix](documents/confusion_matrix.png)

### 3 Specific Wrong Predictions (Fine-Tuned Model)

**1.** *"Working in AI for 8 Years Taught Me Why Both AI Doomers and Optimists Miss the Point."*
- **True:** Project_Development | **Predicted:** Discussion_General (confidence: 0.61)
- **Analysis:** This title describes an opinion essay — no artifact, no system, no implementation. By the label definition and its linguistic profile, it is unambiguously Discussion_General. The Engineering flair labeled it Project_Development, but this is flair noise: the subreddit's Engineering flair is applied inconsistently and sometimes covers opinion/analysis posts. The model correctly matched the text pattern to Discussion_General; the ground truth label is likely wrong. This illustrates a fundamental limit of flair-derived labeling.

**2.** *"Has anyone found a reliable way to enforce strict JSON outputs at scale?"*
- **True:** Project_Development | **Predicted:** Discussion_General (confidence: 0.60)
- **Analysis:** Genuine ambiguity. "Has anyone found" is an interrogative, open-ended pattern that resembles Discussion_General questions in training data. But "enforce strict JSON outputs at scale" is a production engineering problem with a specific technical goal — the kind of thing someone building a system would ask. The model saw the question framing and defaulted to Discussion_General. At 0.60 confidence it was nearly uncertain; the majority-class prior tipped the decision.

**3.** *"AI image detection models?"*
- **True:** Project_Development | **Predicted:** Discussion_General (confidence: 0.62)
- **Analysis:** An extremely short title (four words) that provides almost no signal. It could be asking about what AI image detection models exist (Discussion_General) or looking for a model to use in a specific project (Project_Development). With no vocabulary to pattern-match, the model again defaulted to the majority class. Short titles are a systematic failure mode: the model needs enough vocabulary to identify project-specific framing, and four words provide none.

### Sample Classifications Table (Fine-Tuned Model)

| Post Text | True Label | Predicted Label | Confidence | Correct? |
|---|---|---|---|---|
| "Is your team reviewing AI-generated code?" | Discussion_General | Discussion_General | ~0.92 | ✓ |
| "Has anyone found a reliable way to enforce strict JSON outputs at scale?" | Project_Development | Discussion_General | 0.60 | ✗ |
| "AI image detection models?" | Project_Development | Discussion_General | 0.62 | ✗ |
| "Working in AI for 8 Years Taught Me Why Both AI Doomers and Optimists Miss the Point." | Project_Development | Discussion_General | 0.61 | ✗ |
| "AI Engineer, wants to learn more about Audio related flows, agents, tts, voice cloning and other stuffs in the space. Suggestions please" | Project_Development | Discussion_General | 0.64 | ✗ |

**Correct example explained:** *"Is your team reviewing AI-generated code?"* is a workplace opinion question — no tool names, no system being built, no architecture terms. It matches the Discussion_General vocabulary pattern the model learned from 166 training examples of this class. The model predicted it with high confidence (~0.92) because there is nothing in the title to confuse it with Project_Development language.

---

## Reflection

**What the model learned vs. what I intended:**

I intended a classifier that distinguishes project-oriented content from discussion-oriented content. What the model learned is: predict Discussion_General for everything. With 166 DG examples vs. 27 PD examples in training (86%/14% split), the model never saw enough Project_Development variety to learn a reliable decision boundary. Its fine-tuned accuracy of 85.7% is almost entirely explained by the 36/36 perfect recall on Discussion_General — it classified every single test example as Discussion_General, achieving 0% recall on Project_Development.

The baseline (zero-shot Groq) was more honest about uncertainty: it identified 1 of 6 Project_Development posts correctly (17% recall, F1=0.11). That is still poor, but it is measurably better than 0%. The fine-tuned model's large overall accuracy gain (+23.8 pp over baseline) masked a complete collapse on the minority class — the classic imbalanced-data failure mode where optimizing accuracy on a skewed dataset produces a trivial majority-class predictor.

The root cause is not the model or the training procedure — it is the label source. Using Reddit flairs as proxy labels introduced inconsistency (Engineering flair applied to career posts, opinion essays, and learning questions) and produced a dataset where the minority class was too small and too noisy to teach a reliable boundary. With only 27 PD training examples and several of those likely mislabeled, the model had no chance to learn what Project_Development actually looks like.

---

## Spec Reflection

**Where the spec helped:** Writing the label taxonomy in `planning.md` before data collection forced a clear definition of "project-oriented" before seeing any data. This prevented label drift during the flair-remapping step — whenever an edge case arose (e.g., should a "Discussion" flair post about RAG pipelines be Project_Development?), I had a written definition to consult rather than making ad hoc calls.

**Where implementation diverged from the spec:** The planning document targeted 200+ examples with at least 30% Project_Development (≥60 examples). The final dataset has 276 examples but only 14.1% Project_Development (39 examples) — the minority class target was not met, and all available posts had been collected. The subreddit's natural flair distribution is heavily skewed toward Discussion, which could not be overcome by additional scraping. This divergence from the spec directly caused the model's minority-class failure, which the 30% target was specifically designed to prevent.

---

## AI Usage

**Instance 1 — Designing the flair-to-label remapper**

I gave Claude the full list of unique Reddit flair values observed in the dataset along with my two label definitions from `planning.md`. I asked it to propose a mapping from each flair to one of the two categories. Claude produced an initial mapping that I reviewed entry by entry. I overrode several decisions — for example, Claude initially mapped `Data` to `Discussion_General` because "data" sounds like a general topic, but I changed it to `Project_Development` because posts with this flair were about data pipelines and dataset construction. I also added a `.fillna("Discussion_General")` fallback for unmapped flairs, which Claude had not included.

**Instance 2 — Drafting the Groq classification prompt**

I asked Claude to draft a zero-shot classification prompt for `llama-3.3-70b-versatile`, providing my two label definitions and two examples per label. Claude produced a structured prompt with definitions, examples, and output instructions. I revised it in two ways: (1) I removed a step-by-step reasoning chain Claude had included — it caused the model to output multi-sentence explanations rather than just the label name, breaking the parser; (2) I shortened the examples to match the style and length of actual Reddit titles rather than the longer hypothetical examples Claude generated.

**Annotation assistance disclosure:** The flair-to-label remapping was designed with AI assistance for initial suggestions. All final mapping decisions were reviewed and approved by a human. No posts were labeled entirely by AI without human review. The Groq baseline prompt was drafted with AI assistance and revised by a human before use.

---

## Dataset

| File | Description |
|---|---|
| [`documents/aiengineering_reddit_posts.csv`](documents/aiengineering_reddit_posts.csv) | Labeled dataset — 276 Reddit post titles with flair-derived labels and annotation notes |
| [`documents/evaluation_results.json`](documents/evaluation_results.json) | Accuracy metrics for both models |
| [`documents/confusion_matrix.png`](documents/confusion_matrix.png) | Confusion matrix for fine-tuned model |
| [`documents/ai201_project3_takemeter_starter_clean.ipynb`](documents/ai201_project3_takemeter_starter_clean.ipynb) | Training and evaluation notebook |

---

## Project Structure

```
ai201-project3-takemeter/
├── documents/
│   ├── aiengineering_reddit_posts.csv                # Labeled dataset (276 examples)
│   ├── ai201_project3_takemeter_starter_clean.ipynb  # Training notebook
│   ├── evaluation_results.json                       # Accuracy metrics
│   └── confusion_matrix.png                          # Confusion matrix image
├── planning.md                                       # Design spec (written before data collection)
└── README.md
```

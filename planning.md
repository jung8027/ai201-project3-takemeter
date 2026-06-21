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

## Data Collection Plan

**Source:** Reddit posts from r/aiengineering collected via the public Reddit API. I will use post titles as the primary text feature — titles are short, self-contained, and carry the post's intent without requiring the body.

**Volume target:** 200+ labeled examples, aiming for at least 30% minority class (Project_Development).

**Labeling approach:** Use Reddit post flairs as proxy labels, then consolidate flair variants into the two categories via a remapping table. The subreddit uses flairs like "Discussion", "Engineering", "RAGDiscussion", "Data", etc. I will create a full mapping from each observed flair to one of the two labels. Posts with unmapped or missing flairs will default to `Discussion_General` unless manually reviewed.

**Expected challenges:**
- Short titles (1–5 words) that lack enough signal
- Posts that blend discussion and project content ("I'm building X — is this approach good?")
- Flair inconsistency across users

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

*Document written before data collection. Updated before stretch features.*

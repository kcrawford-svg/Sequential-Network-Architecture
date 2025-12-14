# DBpedia Topic Classification with CNNs and Transformers

This project explores text classification on the DBpedia ontology dataset using a progression of deep learning models. I compare baseline CNN architectures against multiple Transformer-based models, including traditional sinusoidal positional embeddings and Rotary Positional Embeddings (RoPE).

The goal of the project is to understand how different model architectures and positional encoding strategies affect performance on a multi-class topic classification task.

### Why This Matters

This type of text classifier is commonly used in document routing, search systems, and knowledge graph pipelines, where large collections of text must be automatically categorized by topic.

---

## Dataset

This project uses the **DBpedia Ontology Classification Dataset** from Hugging Face. The dataset is constructed from Wikipedia articles and organized into **14 non-overlapping topic classes**.

For this project, I worked with a **balanced subset of 112,000 samples**, with **8,000 examples per class**. Each example consists of a short paragraph of text labeled with its corresponding topic.

**Dataset link:**  
https://huggingface.co/datasets/fancyzhx/dbpedia_14

---

## Models Implemented

### CNN Models

- **Baseline CNN**  
  A simple convolutional neural network used as a reference model.

- **CNN Model B**  
  An improved CNN with additional convolutional layers, normalization, and pooling.

### Transformer Models

- **Transformer A**  
  A single self-attention block with learned token and positional embeddings.

- **Transformer B**  
  Two self-attention blocks with increased embedding size.

- **Transformer B + Learning Rate Scheduler**  
  Same architecture as Transformer B, but trained with an exponential decay learning rate.

- **Transformer with Sinusoidal Positional Embeddings**  
  Uses fixed sinusoidal embeddings added to token embeddings.

- **Transformer with Rotary Positional Embeddings (RoPE)**  
  Applies rotary positional encoding directly to the query and key vectors inside the attention mechanism.

---

## Positional Encoding Experiments

Two positional encoding strategies were compared:

- **Sinusoidal positional embeddings** (from *Attention Is All You Need*)
- **Rotary Positional Embeddings (RoPE)**

RoPE was implemented manually in TensorFlow by rewriting the attention mechanism, since TensorFlow does not expose internal query/key projections in the `MultiHeadAttention` API.

ChatGPT was used **only** to help translate RoPE equations from PyTorch-style references into TensorFlow-compatible code.

---

## Evaluation Metrics

Models were evaluated using:

- **Accuracy**
- **Macro-averaged F1 score**

Macro-F1 was chosen because all classes are evenly distributed and equally important. This metric avoids performance being dominated by any single class and better reflects balanced model behavior.

---

## Results (Macro F1)

| Model | Macro F1 |
|------|----------|
| Baseline CNN | 0.0095 |
| CNN Model B | 0.9558 |
| Transformer A | 0.9820 |
| Transformer B | **0.9847** |
| Transformer B + Scheduler | 0.9846 |
| Transformer + Sinusoidal | 0.9826 |
| Transformer + RoPE | 0.9827 |

Transformer-based models significantly outperform CNNs. Transformer B achieved the highest overall macro-F1 score.

---

## Key Takeaways

- CNNs struggle to capture long-range semantic relationships in text.
- Transformers perform much better on topic classification tasks.
- Adding a second transformer block improves performance but increases training time.
- RoPE and sinusoidal embeddings perform nearly identically on this dataset.
- For this task, architectural capacity mattered more than the choice of positional encoding.

---

## Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- scikit-learn
- Matplotlib
- Hugging Face Datasets

---

## References

- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017).  
  *Attention Is All You Need*. NeurIPS.  
  https://arxiv.org/abs/1706.03762

- Fancyzhx. (2014).  
  *DBpedia Ontology Classification Dataset*. Hugging Face.  
  https://huggingface.co/datasets/fancyzhx/dbpedia_14

- Sharma, P. (2024).  
  *Understanding Rotary Positional Embedding and Implementation*. Medium.  
  https://medium.com/@parulsharmmaa/understanding-rotary-positional-embedding-and-implementation-9f4ad8b03e32

- Larson, E. (n.d.).  
  *Sequence Basics [Experimental]*.  
  https://github.com/eclarson/MachineLearningNotebooks/blob/master/13a.%20Sequence%20Basics%20%5Bexperimental%5D.ipynb

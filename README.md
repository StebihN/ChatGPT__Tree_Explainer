# GPT Tree Explainer

Some machine learning models are very difficult or impossible to understand. The interpretability of models is a metric that indicates how well we can explain the model's decisions. Model quality and interpretability are typically linked in such a way that higher quality models are more difficult to interpret. In such cases, we use techniques to explain the decisions. One of these is the proxy model, which uses the training data and the decisions of an uninterpretable model and extracts patterns from them. The structure of the proxy model must be transparent, so decision trees are an appropriate choice.

However, since decision trees can also be difficult to understand for non-experts, in this volume we will present the operation of the GPTTreeExplainer class, which is designed to easily explain inexplicable models using a proxy decision tree.

The full demonstration of the GPTTreeExplainer class is in the `GPTTreeExplainer_demonstration.ipynb` file.
### 😆 Overview

This assignment explores multi-label emotion classification using the Google GoEmotions dataset. The goal is to build, compare, and evaluate different neural network architectures for predicting emotions from text. 

Models:
- Baseline – BiLSTM model with average pooling
- Attention – Attention-based BiLSTM model with learned token weighting
- Stacked – Stacked BiLSTM model

### Data Sources

[GoEmotions](https://huggingface.co/datasets/google-research-datasets/go_emotions)
- For this implementation, the `Simplified` subset was used
- Dataset is labelled with 27 emotion labels + neutral. Since a single comment can express multiple emotions, labels are represented as multi-hot vectors
- Some limitations out of scope of this implementation are:
  - Class imbalance since some emotions appear much less frequently than others
  - Threshold-based decision making, because each output is predicted independently with sigmoid in the activation layer

### 🤸‍♀️ Model Pipelines
- Baseline
  - Text input
  - Text vectorization
  - Embedding layer
  - Bidirectional LSTM with return_sequences=True
  - GlobalAveragePooling1D
  - Dense(64, ReLU)
  - Dropout
  - Dense(num_classes, sigmoid)

- Attention
  - Text input
  - Text vectorization
  - Embedding layer
  - Bidirectional LSTM with return_sequences=True
  - Dropout
  - AttentionPooling (This layer gives each word in the sentence a score based on how important it is.)
  - Dense(64, ReLU)
  - Dropout
  - Dense(num_classes, sigmoid)
 
- Stacked
  - Text input
  - Text vectorization
  - Embedding layer
  - Bidirectional LSTM with return_sequences=True
  - Bidirectional LSTM
  - Dropout
  - Dense(64, ReLU)
  - Dropout
  - Dense(num_classes, sigmoid)

### Evaluation

The models are evaluated using metrics such as: Accuracy, Precision, Recall, Micro F1, Macro F1

### 🤷‍♀️ Usage
- For local training,
```
pipenv run python ./src/build_and_train_attention.py  
```
- For local evaluation,
```
pipenv run python -m eval.evaluate      

or 

pipenv run python -m eval.evaluate --model-name=attention
```

### Key Findings
- Perfomance
  - The Attention model is the overall top performer, with the highest Macro F1 and the best balance across emotion categories
  - The Baseline model remains competitive, especially for precision, even though it is the simplest model
  - The Stacked model performs the best for peak performance for a few emotions, but it performs the worst for overall balance  (lowest Macro F1)
- Class Imbalance
  - The models' performances are largely influenced by the frequency of the labels in the GoEmotions dataset 
  - For the high-frequency emotions such as gratitude, amusement, and love, the F1 scores are consistently high (~0.75-0.90)
  - For the lower-frequency emotions such as grief, pride, and relief, the F1 scores are close to zero for all models
- Takeaways
  - An attention-based model may be the best choice for real-world applications due to higher recall and better coverage
  - Meaningful improvement will require dataset enhancements, not just architectural changes

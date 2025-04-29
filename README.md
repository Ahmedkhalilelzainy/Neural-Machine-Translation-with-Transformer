# Neural Machine Translation with Transformer

This project implements a Transformer-based model for Neural Machine Translation (NMT) using Python and PyTorch. The model translates French text to English from scratch and is built upon the original Transformer architecture introduced in the paper [Attention is All You Need](https://arxiv.org/abs/1706.03762). This implementation covers all the core components of the Transformer model, including multi-head attention, positional embeddings, and encoder-decoder architecture.

## Features
- **Transformer Architecture**: Custom implementation of the original Transformer architecture.
- **Multi-Head Attention**: Custom attention mechanism to improve model's focus on different parts of the input sequence.
- **Positional Encoding**: Adds positional information to the input embeddings to account for the sequence order.
- **Encoder-Decoder Architecture**: Utilizes both encoder and decoder stacks for sequence-to-sequence translation.
- **BPE Tokenization**: Used to preprocess the French and English text data into subword units.
- **BLEU Score Evaluation**: Model performance evaluated using BLEU scores, which measure the quality of the machine-generated translation against human reference translations.

## Model Architecture

The model consists of the following components:

- **Encoder**: The encoder processes the input sequence and generates key, value, and query vectors that the decoder uses to produce the translation.
- **Decoder**: The decoder takes the encoder’s output and generates the translated sequence.
- **Attention Mechanism**: The multi-head attention mechanism allows the model to focus on different parts of the input sequence for better translation quality.
- **Feedforward Network**: A fully connected feedforward network is used in both the encoder and decoder for nonlinear transformations.

## Results

Reached 56% bleu score.

## Future Improvements
- **Pre-trained Models**: Consider using pre-trained models like BERT or GPT to further enhance performance.
- **Data Augmentation**: Improve the model by augmenting the training dataset with additional parallel corpora.
- **Hyperparameter Tuning**: Explore different hyperparameter settings such as learning rate, batch size, and number of layers to improve the model's translation accuracy.
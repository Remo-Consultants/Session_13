
# SmollM-135M Causal Language Model

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)

This repository contains a clean, from-scratch implementation of a Llama-style causal language model with 135 million parameters. It is designed to match the **SmolLM-2-135M** architecture. The project includes the complete model code, a training script to run on a local GPU, and demonstrates tokenization, training, checkpointing, and inference.

## Features

- **Decoder-Only Transformer**: A standard architecture for modern auto-regressive language models.
- **Grouped-Query Attention (GQA)**: An optimized attention mechanism for reduced memory and computation.
- **Rotary Positional Embeddings (RoPE)**: Effectively captures relative positional information.
- **SwiGLU MLP**: A Swish-Gated Linear Unit for improved performance over standard activations.
- **Flash Attention**: Utilizes `torch.nn.functional.scaled_dot_product_attention` for a highly optimized attention calculation.
- **Mixed-Precision Training**: Supports `bfloat16` for faster training and reduced memory usage.

## Project Structure

```
.
├── SmolLM2-135M/       # Pre-trained tokenizer and model weights
├── checkpoints/        # Saved model checkpoints during training
├── model.py            # Main script for training the model from scratch
├── smollm_demo.py      # Script to run inference with a trained model
├── app.py              # A simple Gradio web interface for the model
├── input-1.txt         # Sample text file for training
└── README.MD           # You are here!
```

## Getting Started

Follow these instructions to set up the environment and start using the model.

### Prerequisites

- Python 3.9+
- PyTorch 2.0+ (with CUDA support for GPU training)
- Hugging Face Transformers and Tokenizers
- Gradio (for the web UI)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/SmollM135M.git
    cd SmollM135M
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install torch transformers tokenizers gradio
    ```

### Running Inference

To see the model in action, run the demo script, which loads a saved checkpoint and generates text.

```bash
python smollm_demo.py --prompt "The quick brown fox" --checkpoint_path "checkpoints/model_final_5000.pt"
```

### Running the Web App

Launch a simple web interface to interact with the model.

```bash
python app.py
```

### Training the Model

To train the model from scratch, run the `model.py` script. The script is pre-configured to train for 5000 iterations and save checkpoints.

```bash
python model.py
```

The script will automatically use a CUDA-enabled GPU if available.

## Model Architecture

The model is a decoder-only transformer, composed of a stack of identical decoder blocks that process input token embeddings to predict the next token in a sequence.

The key components of each block are:
- **RMSNorm**: A computationally efficient normalization technique.
- **Grouped-Query Attention (GQA)**: Reduces the memory footprint and computational cost of attention.
- **Rotary Positional Embeddings (RoPE)**: Applies positional embeddings directly to queries and keys.
- **SwiGLU MLP**: A Swish-Gated Linear Unit in the feed-forward layer.
- **Flash Attention**: A highly optimized kernel for scaled dot-product attention.

The overall data flow is as follows:

`Token IDs -> Embedding Layer -> N x Decoder Blocks -> Final RMSNorm -> LM Head -> Logits`

## Model Configuration & 135M Parameters

The model's size is determined by the `ModelConfig` class in `model.py`.

| Parameter | Value | Description |
| :--- | :--- | :--- |
| `hidden_size` | 576 | The dimensionality of the embedding and hidden states. |
| `intermediate_size` | 1536 | The inner dimension of the MLP layer. |
| `num_hidden_layers` | 30 | The number of stacked DecoderLayer blocks. |
| `num_attention_heads` | 9 | The number of query heads in the attention mechanism. |
| `num_key_value_heads` | 3 | The number of key/value heads for GQA. |
| `vocab_size` | 49152 | The number of unique tokens in the vocabulary. |
| `max_position_embeddings`| 8192 | The maximum sequence length the model can process. |

#### Parameter Calculation
The "135M" parameter count is an approximation derived from these configuration values.

- **Token Embeddings & LM Head**: `vocab_size * hidden_size` = 49152 * 576 ≈ **28.3M** (weights are tied).
- **Parameters per Decoder Block**:
  - **Attention (Q,K,V,O)**: ~0.88M
  - **MLP (Gate, Up, Down)**: ~2.65M
  - **Total per Block**: ~3.53M
- **Total Parameters**: (`Params per Block * Num Layers`) + `Embedding Params` = (3.53M * 30) + 28.3M ≈ **134.2M**

This calculation confirms the configuration closely matches the target of 135 million parameters.

## How it Works: From Input to Prediction

The process of predicting the next token involves several sequential steps:

1.  **Tokenization**: Raw text is converted into a sequence of integer IDs using a Hugging Face `AutoTokenizer`.
2.  **Embedding**: The token IDs are mapped to high-dimensional vectors via an `nn.Embedding` layer.
3.  **Processing through Decoder Blocks**: The sequence of vectors is passed through 30 decoder blocks, where contextual understanding is built using self-attention and MLP layers. A causal mask ensures a token can only see previous tokens.
4.  **Final Projection (LM Head)**: The vector corresponding to the final input token is projected into a large vector with `vocab_size` dimensions.
5.  **Logits to Prediction**: This final vector contains logits. The index with the highest value (`argmax`) corresponds to the most likely next token ID.
6.  **Decoding**: The predicted token ID is converted back into a human-readable string.

## Training Logs

The model is trained using `model.py`. The script is configured to run for 5000 iterations, save checkpoints every 500 steps, and resume training. Below is a sample of the training output.

```
Using device: cuda
Initializing model in float32 for mixed-precision training.
Total training sequences generated: 9

--- Starting Initial Training ---
Iteration 100/5000, Loss: 31.4173
...
Iteration 5000/5000, Loss: 0.0092
Checkpoint saved at checkpoints\model_checkpoint_5000.pt
Final checkpoint saved at checkpoints\model_final_5000.pt

--- Resuming Training from Final Checkpoint for 50 more iterations ---
Resuming from iteration: 5000
Additional Iteration 5010, Loss: 0.0015
...
Additional Iteration 5050, Loss: 0.0004

--- Training complete. Checking output with the latest model state ---
Input text: 'The quick brown fox'
Predicted next token: ' fox'
```

## Contributing

Contributions are welcome! If you have suggestions for improvements or find any issues, please feel free to open an issue or submit a pull request.

1.  **Fork** the repository.
2.  Create a new branch (`git checkout -b feature/your-feature-name`).
3.  Make your changes and **commit** them (`git commit -m 'Add some feature'`).
4.  **Push** to the branch (`git push origin feature/your-feature-name`).
5.  Open a **Pull Request**.
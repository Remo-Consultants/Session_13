SmollM-135M Causal Language Model
This repository contains a clean, from-scratch implementation of a Llama-style causal language model with 135 million parameters, configured to match the SmolLM-2-135M architecture. The project includes the complete model code, a training script to run on a local GPU, and demonstrates how to handle tokenization, training, checkpointing, and inference.
1. Model Architecture
The model is a decoder-only transformer, which is standard for modern auto-regressive language models like LLaMA and GPT. It is composed of a stack of identical decoder blocks that process input token embeddings to predict the next token in a sequence.
The key components of each block are:
RMSNorm: A computationally efficient normalization technique used before the attention and MLP layers, as proposed in the LLaMA paper.
Grouped-Query Attention (GQA): An optimized attention mechanism that uses fewer key/value heads than query heads. This significantly reduces the memory footprint and computational cost of the attention calculation without a major loss in performance.
Rotary Positional Embeddings (RoPE): Instead of adding positional information at the beginning, RoPE applies positional embeddings directly to the queries and keys within the attention mechanism. This is highly effective for capturing relative positional information.
SwiGLU MLP: A Swish-Gated Linear Unit is used as the activation function in the feed-forward MLP layer, which has been shown to improve performance over standard ReLU or GeLU activations.
Flash Attention: The core attention calculation uses torch.nn.functional.scaled_dot_product_attention, a highly optimized PyTorch function that leverages Flash Attention when available. It handles the computationally intensive dot-product, scaling, and masking (including causal masking) in a single, efficient kernel.
The overall data flow is as follows:
Token IDs -> Embedding Layer -> N x Decoder Blocks -> Final RMSNorm -> LM Head -> Logits
2. Model Configuration & 135M Parameters
The model's size is determined by the specific configuration set in the ModelConfig class inside model.py. These parameters define the dimensions and depth of the network.
Parameter	Value	Description
hidden_size	576	The dimensionality of the embedding and hidden states.
intermediate_size	1536	The inner dimension of the MLP (feed-forward) layer.
num_hidden_layers	30	The number of stacked DecoderLayer blocks.
num_attention_heads	9	The number of query heads in the attention mechanism.
num_key_value_heads	3	The number of key/value heads for GQA.
vocab_size	49152	The number of unique tokens in the tokenizer vocabulary.
max_position_embeddings	8192	The maximum sequence length the model can process.
Parameter Calculation
The "135M" parameter count is an approximation derived from these configuration values. The vast majority of parameters are in the embedding/LM head, the attention blocks, and the MLP blocks.
Token Embeddings & LM Head: vocab_size * hidden_size = 49152 * 576 ≈ 28.3M (The LM head weights are tied to the embedding layer, so we only count this once).
Parameters per Decoder Block:
Attention (Q,K,V,O projections):
Q-proj: hidden_size * hidden_size = 576 * 576 = 331,776
K-proj: hidden_size * (n_kv_heads * head_dim) = 576 * (3 * 64) = 110,592
V-proj: hidden_size * (n_kv_heads * head_dim) = 576 * (3 * 64) = 110,592
O-proj: hidden_size * hidden_size = 576 * 576 = 331,776
Total Attention: ~0.88M
MLP (Gate, Up, Down projections):
Gate-proj: hidden_size * intermediate_size = 576 * 1536 = 884,736
Up-proj: hidden_size * intermediate_size = 576 * 1536 = 884,736
Down-proj: intermediate_size * hidden_size = 1536 * 576 = 884,736
Total MLP: ~2.65M
Normalization: 2 * RMSNorm layers per block, each with hidden_size parameters = 2 * 576 = 1,152.
Total per Block: ~0.88M + ~2.65M = ~3.53M
Total Parameters: (Params per Block * Num Layers) + Embedding Params = (3.53M * 30) + 28.3M ≈ 105.9M + 28.3M ≈ 134.2M
This calculation confirms that the configuration closely matches the target of 135 million parameters.
3. Training Logs
The model is trained using the provided model.py script. The script is configured to run for 5000 iterations, save checkpoints every 500 steps, and then resume for an additional 50 steps.
Below is an example of the expected output. You can replace this with your actual logs after running the training.

Using device: cuda
Initializing model in float32 for mixed-precision training.
Total training sequences generated: 9

--- Starting Initial Training ---
Iteration 100/5000, Loss: 31.4173
Iteration 200/5000, Loss: 21.7289
Iteration 300/5000, Loss: 14.5357
Iteration 400/5000, Loss: 11.9046
Iteration 500/5000, Loss: 9.9928
Checkpoint saved at checkpoints\model_checkpoint_500.pt
Iteration 600/5000, Loss: 9.4807
Iteration 700/5000, Loss: 7.9402
Iteration 800/5000, Loss: 7.1853
Iteration 900/5000, Loss: 6.5923
Iteration 1000/5000, Loss: 7.8161
Checkpoint saved at checkpoints\model_checkpoint_1000.pt
Iteration 1100/5000, Loss: 6.0631
Iteration 1200/5000, Loss: 5.7473
Iteration 1300/5000, Loss: 5.8958
Iteration 1400/5000, Loss: 5.4149
Iteration 1500/5000, Loss: 5.2917
Checkpoint saved at checkpoints\model_checkpoint_1500.pt
Iteration 1600/5000, Loss: 4.3931
Iteration 1700/5000, Loss: 5.2131
Iteration 1800/5000, Loss: 3.2277
Iteration 1900/5000, Loss: 2.7828
Iteration 2000/5000, Loss: 1.7538
Checkpoint saved at checkpoints\model_checkpoint_2000.pt
Iteration 2100/5000, Loss: 0.8362
Iteration 2200/5000, Loss: 0.1121
Iteration 2300/5000, Loss: 0.0646
Iteration 2400/5000, Loss: 0.0221
Iteration 2500/5000, Loss: 0.0357
Checkpoint saved at checkpoints\model_checkpoint_2500.pt
Iteration 2600/5000, Loss: 0.0246
Iteration 2700/5000, Loss: 0.0316
Iteration 2800/5000, Loss: 0.0253
Iteration 2900/5000, Loss: 0.0379
Iteration 3000/5000, Loss: 0.0382
Checkpoint saved at checkpoints\model_checkpoint_3000.pt
Iteration 3100/5000, Loss: 0.0306
Iteration 3200/5000, Loss: 0.0276
Iteration 3300/5000, Loss: 0.0096
Iteration 3400/5000, Loss: 0.0376
Iteration 3500/5000, Loss: 0.0244
Checkpoint saved at checkpoints\model_checkpoint_3500.pt
Iteration 3600/5000, Loss: 0.0289
Iteration 3700/5000, Loss: 0.0045
Iteration 3800/5000, Loss: 0.0085
Iteration 3900/5000, Loss: 0.0078
Iteration 4000/5000, Loss: 0.0041
Checkpoint saved at checkpoints\model_checkpoint_4000.pt
Iteration 4100/5000, Loss: 0.0083
Iteration 4200/5000, Loss: 0.0014
Iteration 4300/5000, Loss: 0.0002
Iteration 4400/5000, Loss: 0.0004
Iteration 4500/5000, Loss: 0.0001
Checkpoint saved at checkpoints\model_checkpoint_4500.pt
Iteration 4600/5000, Loss: 0.0004
Iteration 4700/5000, Loss: 0.0092
Iteration 4800/5000, Loss: 0.0088
Iteration 4900/5000, Loss: 0.0042
Iteration 5000/5000, Loss: 0.0092
Checkpoint saved at checkpoints\model_checkpoint_5000.pt
Final checkpoint saved at checkpoints\model_final_5000.pt

--- Resuming Training from Final Checkpoint for 50 more iterations ---
Resuming from iteration: 5000
Additional Iteration 5010, Loss: 0.0015
Additional Iteration 5020, Loss: 0.0001
Additional Iteration 5030, Loss: 0.0002
Additional Iteration 5040, Loss: 0.0000
Additional Iteration 5050, Loss: 0.0004

--- Training complete. Checking output with the latest model state ---
Input text: 'The quick brown fox'
Predicted next token: ' fox'

4. How it Works: From Input to Prediction
The process of predicting the next token from a piece of text involves several sequential steps:
Tokenization: The process begins with raw text, such as the content of input-1.txt. A Hugging Face AutoTokenizer is used to convert this string of characters into a sequence of integer IDs. Each ID corresponds to a specific token (a word or sub-word) in the tokenizer's vocabulary.
Embedding: The sequence of token IDs is fed into the model's nn.Embedding layer. This layer acts as a lookup table, converting each integer ID into a high-dimensional vector (of size hidden_size=576). This vector is the initial representation of the token.
Processing through Decoder Blocks: This sequence of vectors is then passed through the stack of 30 DecoderLayer blocks. Inside each block:
The vectors are first normalized by RMSNorm.
The normalized vectors are then processed by the self-attention mechanism. For each token, the attention layer calculates how much it should "pay attention to" every other token that came before it in the sequence. The built-in causal mask ensures that a token at position i can only see tokens from 0 to i, not future ones. This is where the model builds contextual understanding.
The output of the attention layer is added back to the original input (a residual connection) and then passed to another RMSNorm layer.
Finally, the normalized output is processed by the MLP (feed-forward) network, which performs further non-linear transformations on the data. The result is again added back via a residual connection.
Final Projection (LM Head): After passing through all 30 blocks, the final sequence of processed vectors is normalized one last time. The vector corresponding to the last token in the input sequence is then fed into the lm_head layer. This is a linear layer that projects the hidden_size (576) dimensional vector into a very large vector with vocab_size (49152) dimensions.
Logits to Prediction: This final, large vector contains the logits. Each element in this vector corresponds to a token in the vocabulary. The higher the logit value, the more likely the model thinks that token is the next one in the sequence. For inference, we simply find the index with the highest value (argmax).
Decoding: The resulting index is a token ID. The tokenizer's decode method is used to convert this integer ID back into a human-readable string, giving us the predicted next word.
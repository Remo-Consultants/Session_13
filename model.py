import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
import os
import math
from torch.amp import autocast, GradScaler # Use modern torch.amp API

# -----------------------------------------------------------------
# Model Components from smollm2_model.py
# -----------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        rms = x.pow(2).mean(-1, keepdim=True).sqrt()
        x = x / (rms + self.eps)
        return self.weight * x

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: int = 100_000):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len: int, device: torch.device):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos().unsqueeze(0).unsqueeze(0)
        sin = emb.sin().unsqueeze(0).unsqueeze(0)
        return cos, sin

def apply_rotary_pos_emb(q, k, cos, sin):
    q_rot = (q * cos) + (torch.stack([-q[..., 1::2], q[..., ::2]], dim=-1).flatten(-2) * sin)
    k_rot = (k * cos) + (torch.stack([-k[..., 1::2], k[..., ::2]], dim=-1).flatten(-2) * sin)
    return q_rot, k_rot

class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_heads = cfg.num_attention_heads
        self.n_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.hidden_size // cfg.num_attention_heads
        self.n_groups = self.n_heads // self.n_kv_heads

        self.q_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=cfg.attention_bias)
        self.k_proj = nn.Linear(cfg.hidden_size, cfg.num_key_value_heads * self.head_dim, bias=cfg.attention_bias)
        self.v_proj = nn.Linear(cfg.hidden_size, cfg.num_key_value_heads * self.head_dim, bias=cfg.attention_bias)
        self.o_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=cfg.attention_bias)

        self.rope = RotaryEmbedding(self.head_dim, theta=cfg.rope_theta)
        self.attn_dropout = nn.Dropout(cfg.attention_dropout)

    def forward(self, x, attention_mask=None):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if self.n_groups > 1:
            k = k.repeat_interleave(self.n_groups, dim=1)
            v = v.repeat_interleave(self.n_groups, dim=1)

        cos, sin = self.rope(T, x.device)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Use built-in Flash Attention
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=True # Built-in causal masking
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(attn_output)

class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))

class DecoderLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.self_attn = Attention(cfg)
        self.mlp = MLP(cfg)
        self.norm_1 = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.norm_2 = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

    def forward(self, x, attention_mask=None):
        h = self.norm_1(x)
        h = self.self_attn(h, attention_mask)
        x = x + h
        h = self.norm_2(x)
        h = self.mlp(h)
        return x + h

class LlamaModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size, padding_idx=cfg.pad_token_id)
        self.layers = nn.ModuleList([DecoderLayer(cfg) for _ in range(cfg.num_hidden_layers)])
        self.norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

    def forward(self, input_ids, attention_mask=None):
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x, attention_mask)
        return self.norm(x)

class LlamaForCausalLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = LlamaModel(cfg)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids, attention_mask=None, labels=None):
        hidden = self.model(input_ids, attention_mask)
        logits = self.lm_head(hidden)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.cfg.vocab_size), shift_labels.view(-1))
        return {"logits": logits, "loss": loss}

# Configuration class to mimic Hugging Face's PretrainedConfig
class ModelConfig:
    def __init__(self):
        self.architectures = ["LlamaForCausalLM"]
        self.attention_bias = False
        self.attention_dropout = 0.0
        self.bos_token_id = 0
        self.eos_token_id = 0
        self.hidden_act = "silu"
        self.hidden_size = 576
        self.intermediate_size = 1536
        self.max_position_embeddings = 8192
        self.num_attention_heads = 9
        self.num_hidden_layers = 30
        self.num_key_value_heads = 3
        self.rms_norm_eps = 1e-05
        self.rope_theta = 100000
        self.vocab_size = 49152
        self.pad_token_id = 0

# -----------------------------------------------------------------
# Training Loop
# -----------------------------------------------------------------

if __name__ == "__main__":
    # Setup device
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_type)
    print(f"Using device: {device}")

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained('./SmolLM2-135M')
    # Update pad_token_id in config if tokenizer has it defined differently
    model_config = ModelConfig()
    if tokenizer.pad_token_id is not None:
        model_config.pad_token_id = tokenizer.pad_token_id
    else:
        # If no pad token, use eos token for padding
        tokenizer.pad_token = tokenizer.eos_token
        model_config.pad_token_id = tokenizer.eos_token_id
    
    # Read input from input-1.txt
    with open('input-1.txt', 'r', encoding='utf-8') as f:
        text_input = f.read()

    # Initialize the model
    # We will use float16 for autocast, so the model can be in float32
    print("Initializing model in float32 for mixed-precision training.")
    model = LlamaForCausalLM(model_config).to(device)


    # Define optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Initialize GradScaler for mixed precision training
    scaler = GradScaler(enabled=(device_type == 'cuda'))

    # Create checkpoints directory
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # --- Training Loop Setup ---
    num_iterations = 5000
    log_interval = 100
    checkpoint_interval = 500
    batch_size = 1 # Keep at 1 for memory constraints

    # Tokenize and prepare data
    encoded_input = tokenizer(text_input, return_tensors="pt", truncation=True, max_length=model_config.max_position_embeddings * 10).input_ids[0]
    
    seq_len = model_config.max_position_embeddings
    input_sequences = [encoded_input[i:i+seq_len+1] for i in range(0, len(encoded_input) - seq_len - 1, seq_len)]

    if not input_sequences:
        print("Warning: Input text is too short to form sequences of required length. Training will not proceed.")
    else:
        all_data = torch.stack(input_sequences)
        print(f"Total training sequences generated: {len(all_data)}")

        print("\n--- Starting Initial Training ---")
        for iteration in range(num_iterations):
            idx = iteration % len(all_data)
            batch = all_data[idx].unsqueeze(0) # Guaranteed shape: (1, seq_len+1)
            
            input_ids = batch[:, :-1].to(device)
            labels = batch[:, 1:].to(device)
            
            # The built-in scaled_dot_product_attention handles causal masking.
            # We only need to provide a mask for padding if it exists.
            # In this setup with fixed sequence lengths, a padding mask is not needed.
            attention_mask = None 

            optimizer.zero_grad(set_to_none=True)

            # Use torch.float16 for autocast, as it's fully supported by GradScaler
            with autocast(device_type=device_type, dtype=torch.float16):
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs["loss"]

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if (iteration + 1) % log_interval == 0:
                print(f"Iteration {iteration + 1}/{num_iterations}, Loss: {loss.item():.4f}")

            if (iteration + 1) % checkpoint_interval == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f"model_checkpoint_{iteration + 1}.pt")
                torch.save({'iteration': iteration + 1, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scaler_state_dict': scaler.state_dict()}, checkpoint_path)
                print(f"Checkpoint saved at {checkpoint_path}")

        # Save final checkpoint
        final_checkpoint_path = os.path.join(checkpoint_dir, "model_final_5000.pt")
        torch.save({'iteration': num_iterations, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scaler_state_dict': scaler.state_dict()}, final_checkpoint_path)
        print(f"Final checkpoint saved at {final_checkpoint_path}")

        # --- Resume Training ---
        print("\n--- Resuming Training from Final Checkpoint for 50 more iterations ---")
        loaded_checkpoint = torch.load(final_checkpoint_path)
        model.load_state_dict(loaded_checkpoint['model_state_dict'])
        optimizer.load_state_dict(loaded_checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(loaded_checkpoint['scaler_state_dict'])
        start_iteration = loaded_checkpoint['iteration']
        print(f"Resuming from iteration: {start_iteration}")

        additional_iterations = 50
        for iteration in range(start_iteration, start_iteration + additional_iterations):
            idx = iteration % len(all_data)
            batch = all_data[idx].unsqueeze(0)
            
            input_ids = batch[:, :-1].to(device)
            labels = batch[:, 1:].to(device)
            attention_mask = None

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device_type, dtype=torch.float16):
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs["loss"]

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if (iteration + 1) % (additional_iterations // 5) == 0 or iteration == start_iteration + additional_iterations -1:
                print(f"Additional Iteration {iteration + 1}, Loss: {loss.item():.4f}")

        print("\n--- Training complete. Checking output with the latest model state ---")

        # --- Check Output ---
        inference_text = "The quick brown fox"
        inference_inputs = tokenizer(inference_text, return_tensors="pt").to(device)
        
        model.eval()
        with torch.no_grad():
            with autocast(device_type=device_type, dtype=torch.float16):
                inference_outputs = model(inference_inputs["input_ids"])
            logits = inference_outputs["logits"].to(torch.float32)
            predicted_token_id = torch.argmax(logits[:, -1, :], dim=-1)
            predicted_token = tokenizer.decode(predicted_token_id)
            print(f"Input text: '{inference_text}'")
            print(f"Predicted next token: '{predicted_token}'")
        model.train()
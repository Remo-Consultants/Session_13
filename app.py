import gradio as gr
import torch
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download

# Import the model architecture from your model.py file
from model import LlamaForCausalLM, ModelConfig

# --- Configuration ---
# Point to your Hugging Face Model Hub repository
MODEL_REPO_ID = "DineshSundaram/smollm2_135M"
CHECKPOINT_FILENAME = "model_final_5000.pt"

# Use a valid public tokenizer from the Hub to fix the error
TOKENIZER_PATH = "gpt2"
MAX_SEQUENCE_LENGTH = 8192  # Should match the model's max_position_embeddings

# --- Global Variables ---
model = None
tokenizer = None
device = None

# --- Model Loading ---
def load_model_and_tokenizer():
    """
    Loads the model and tokenizer into global variables.
    This is done once when the app starts.
    """
    global model, tokenizer, device

    # Set up device
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_type)
    print(f"Using device: {device}")

    # Load tokenizer from the Hub
    try:
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print("Tokenizer pad_token set to eos_token.")
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

    # Load model configuration
    model_config = ModelConfig()
    model_config.pad_token_id = tokenizer.pad_token_id
    # Adjust vocab size to match the new tokenizer
    model_config.vocab_size = tokenizer.vocab_size 

    # Instantiate the model
    # Use float16 for inference as it's faster and uses less memory
    if device_type == "cuda" and torch.cuda.is_bf16_supported():
        print("Initializing model in bfloat16 for inference.")
        model = LlamaForCausalLM(model_config).to(device, dtype=torch.bfloat16)
    else:
        print("Initializing model in float32 for inference.")
        model = LlamaForCausalLM(model_config).to(device)
    
    # Download and load the checkpoint from the Hub
    print(f"Downloading checkpoint '{CHECKPOINT_FILENAME}' from repo '{MODEL_REPO_ID}'")
    try:
        checkpoint_path = hf_hub_download(
            repo_id=MODEL_REPO_ID,
            filename=CHECKPOINT_FILENAME
        )
        print(f"Loading checkpoint from downloaded file: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Adjust for potential vocab size mismatch between training and new tokenizer
        original_vocab_size = checkpoint['model_state_dict']['lm_head.weight'].shape[0]
        if original_vocab_size != model_config.vocab_size:
            print(f"Warning: Vocab size mismatch. Trained: {original_vocab_size}, Tokenizer: {model_config.vocab_size}. Adjusting model head.")
            model.resize_token_embeddings(model_config.vocab_size)

        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print("Model state loaded successfully.")
    except Exception as e:
        print(f"An error occurred while downloading or loading the model state: {e}")
        # Create a dummy model so the app doesn't crash on startup
        model = LlamaForCausalLM(model_config).to(device)

    # Set the model to evaluation mode
    model.eval()
    print("Model and tokenizer loaded.")


# --- Inference Function ---
def predict(input_text, max_new_tokens=20, temperature=0.7):
    """
    The main function that runs inference and is called by Gradio.
    """
    global model, tokenizer, device
    
    if model is None or tokenizer is None:
        return "Error: Model or tokenizer not loaded. Please check the console logs."

    # Tokenize the input text
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

    # Generate tokens one by one (auto-regressive decoding)
    generated_ids = input_ids
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # If the sequence gets too long, truncate it
            if generated_ids.shape[1] > MAX_SEQUENCE_LENGTH:
                generated_ids = generated_ids[:, -MAX_SEQUENCE_LENGTH:]

            # Get logits from the model
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16):
                 outputs = model(generated_ids)
            
            logits = outputs['logits']
            
            # Get the logits for the very last token
            next_token_logits = logits[:, -1, :]

            # Apply temperature scaling
            if temperature > 0:
                next_token_logits = next_token_logits / temperature
                # Apply softmax to get probabilities
                probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                # Sample from the distribution
                next_token_id = torch.multinomial(probs, num_samples=1)
            else:
                # Use greedy decoding if temperature is 0
                next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)


            # Append the new token to the sequence
            generated_ids = torch.cat((generated_ids, next_token_id), dim=1)

            # Stop if the end-of-sequence token is generated
            if next_token_id.item() == tokenizer.eos_token_id:
                break
    
    # Decode the generated token IDs back to a string
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return generated_text


# --- Main execution block ---
if __name__ == "__main__":
    # Load the model and tokenizer on app startup
    load_model_and_tokenizer()

    # Create and launch the Gradio interface
    demo = gr.Interface(
        fn=predict,
        inputs=[
            gr.Textbox(lines=5, label="Input Text", placeholder="Type your prompt here..."),
            gr.Slider(minimum=5, maximum=100, value=20, step=1, label="Max New Tokens"),
            gr.Slider(minimum=0.0, maximum=1.5, value=0.7, step=0.1, label="Temperature (0 for deterministic)"),
        ],
        outputs=gr.Textbox(lines=5, label="Generated Text"),
        title="SmollM-135M Language Model",
        description="This is a demo of a 135M parameter Llama-style model. "
                    "It was trained from scratch and is now running with a standard GPT-2 tokenizer.",
    
    )

    demo.launch()
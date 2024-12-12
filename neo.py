import streamlit as st
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
import torch

# Check device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load the tokenizer from the base model
base_model_name = "EleutherAI/gpt-neo-125M"
tokenizer = GPT2Tokenizer.from_pretrained(base_model_name)

# Ensure tokenizer compatibility
tokenizer.pad_token = tokenizer.eos_token

# Load the pretrained fine-tuned model
pretrained_model_name = "AmmarA22/gptneo-wikitext-quantized"
model = GPTNeoForCausalLM.from_pretrained(pretrained_model_name).to(device)

# Function for inference
def infer_with_finetuned_gpt_neo(prompt, model, tokenizer, max_length=200):
    """
    Generate a response from the fine-tuned GPT-Neo model given a prompt.
    """
    # Tokenize the input
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Generate text
    outputs = model.generate(
        inputs["input_ids"],
        max_length=max_length,
        num_return_sequences=1,
        temperature=0.7,  # Controls randomness
        top_p=0.9,  # Top-p sampling for diverse outputs
        do_sample=True
    )

    # Decode the output to a human-readable string
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Streamlit UI
def main():
    st.title("Fine-Tuned GPT-Neo Text Generator")
    st.write("This application uses a fine-tuned GPT-Neo model to generate responses based on user prompts.")

    # Text input for the prompt
    prompt = st.text_area("Enter your prompt:", placeholder="Type something to generate a response...", height=150)

    # Slider for max length
    max_length = st.slider("Maximum Length of Response", min_value=50, max_value=500, value=200, step=10)

    # Button to generate response
    if st.button("Generate Response"):
        if prompt.strip():
            with st.spinner("Generating response..."):
                response = infer_with_finetuned_gpt_neo(prompt, model, tokenizer, max_length)
            st.subheader("Generated Response:")
            st.write(response)
        else:
            st.error("Please enter a prompt to generate a response.")

if __name__ == "__main__":
    main()

import streamlit as st
import requests
from dotenv import load_dotenv
import os
from transformers import pipeline

# Load environment variables from .env file
load_dotenv()

# Retrieve API key from environment variables
HF_API_KEY = os.getenv('HF_API_KEY2')

# Image Generation Function
def generate_image(prompt):
    """Generate image using Stable Diffusion 3.5-large via Hugging Face API."""
    api_url = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-3.5-large"
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    payload = {"inputs": prompt}
    
    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=60)
        if response.status_code == 200:
            image_path = "generated_image.png"
            with open(image_path, "wb") as f:
                f.write(response.content)
            return image_path
        else:
            st.error(f"Image API error: {response.status_code} - {response.text[:100]}...")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Image generation failed: {e}")
        return None

# Chat Response Functions
@st.cache_resource
def load_local_chat_model():
    """Load a local DistilGPT-2 model as a fallback."""
    try:
        st.write("Loading local DistilGPT-2 model (one-time download, ~300 MB)...")
        model = pipeline('text-generation', model='distilgpt2')
        st.write("Local model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Failed to load local model: {e}")
        return None

def generate_chat_response(prompt):
    """Generate response using Llama 3.1 API, fall back to local DistilGPT-2."""
    # Try Hugging Face Router API first
    api_url = "https://router.huggingface.co/featherless-ai/v1/completions"
    headers = {
        "Authorization": f"Bearer {HF_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "meta-llama/Meta-Llama-3.1-8B",
        "prompt": prompt
    }
    
    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=30)
        if response.status_code == 200:
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["text"]
            return "No response generated"
        else:
            st.error(f"Chat API error: {response.status_code} - {response.text[:100]}...")
    except requests.exceptions.RequestException as e:
        st.error(f"Chat API request failed: {e}")

    # Fallback to local model if API fails
    st.write("API unavailable, switching to local DistilGPT-2 model...")
    if 'local_chat_model' not in st.session_state:
        st.session_state.local_chat_model = load_local_chat_model()
    
    if st.session_state.local_chat_model:
        try:
            response = st.session_state.local_chat_model(prompt, max_length=100, num_return_sequences=1)
            return response[0]["generated_text"]
        except Exception as e:
            st.error(f"Local chat generation failed: {e}")
            return "Error occurred while generating response locally."
    else:
        return "No response available (API down and local model failed)."

# Streamlit Interface
st.title("AI Image Generator & Chatbot")
tab1, tab2 = st.tabs(["Image Generator", "Chatbot"])

with tab1:
    st.header("Generate Images from Text")
    image_prompt = st.text_input("Enter a prompt for image generation:", "A futuristic cityscape at night")
    if st.button("Generate Image"):
        with st.spinner("Generating image (may take up to 30 seconds)..."):
            image_result = generate_image(image_prompt)
            if image_result and os.path.exists(image_result):
                st.image(image_result, caption=f"Generated Image: {image_prompt}")
            else:
                st.error("Failed to generate image.")

with tab2:
    st.header("Chat with AI")
    chat_prompt = st.text_area("Enter your chat prompt:", "Explain artificial intelligence in simple terms.")
    if st.button("Get Response"):
        with st.spinner("Generating response..."):
            response = generate_chat_response(chat_prompt)
            st.write("**AI Response:**")
            st.write(response)

st.sidebar.markdown("### About")
st.sidebar.info("This app uses Hugging Face’s Stable Diffusion for images and Flan-T5 for chat, with a local DistilGPT-2 fallback.")
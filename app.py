import streamlit as st
import torch
import os
from PIL import Image

# Import our inference functions
from inference.chat import load_vlm, chat_with_image

# --- Page Config ---
st.set_page_config(
    page_title="RS-VLM | Remote Sensing AI",
    page_icon="🌍",
    layout="centered"
)

# --- Load Model (Cached) ---
@st.cache_resource
def init_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = r"training\checkpoints\model_phase3_final.pt"
    
    if not os.path.exists(checkpoint_path):
        st.error(f"Checkpoint not found at {checkpoint_path}. Please check your path.")
        st.stop()
        
    return load_vlm(checkpoint_path, device)

# --- UI Header ---
st.title("🌍 RS-VLM: Remote Sensing Vision-Language Model")
st.markdown("""
*An end-to-end Vision-Language Model trained from scratch to analyze satellite imagery.*
Upload a satellite image, and ask the AI what it sees!
""")

# Load the model behind the scenes
with st.spinner("Loading AI Model into Memory... (This takes a few seconds)"):
    model = init_model()

# --- Sidebar / Upload ---
st.sidebar.header("1. Upload Image")
uploaded_file = st.sidebar.file_uploader("Choose a satellite image...", type=["jpg", "jpeg", "png"])

st.sidebar.markdown("---")
st.sidebar.header("About this project")
st.sidebar.markdown("""
Built by Aliasghar J.\n
**Architecture:**
- **Vision:** ViT (MAE Pretrained)
- **Bridge:** MLP Projector
- **Brain:** TinyLlama-1.1B (LoRA fine-tuned)
""")

# --- Main Interface ---
if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Satellite Image", use_column_width=True)
    
    # Save the uploaded file temporarily so our chat_with_image function can read it
    temp_img_path = "temp_upload.jpg"
    image.save(temp_img_path)
    
    # Text input for the user's question
    st.markdown("### Ask the AI")
    question = st.text_input("Enter your question:", value="What type of land use is shown in this satellite image?")
    
    if st.button("Analyze Image", type="primary"):
        with st.spinner("Analyzing spectral data and generating response..."):
            try:
                # Run Inference
                answer = chat_with_image(
                    model=model, 
                    image_path=temp_img_path, 
                    question=question,
                    gsd=10.0
                )
                
                st.success("Analysis Complete!")
                st.markdown(f"**🤖 Assistant:** {answer}")
                
            except Exception as e:
                st.error(f"An error occurred during generation: {str(e)}")
            
            finally:
                # Clean up the temp file
                if os.path.exists(temp_img_path):
                    os.remove(temp_img_path)
else:
    st.info("👈 Please upload an image from the sidebar to begin.")

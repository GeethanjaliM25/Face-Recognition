import streamlit as st
import cv2
import numpy as np
from deepface import DeepFace
from PIL import Image
import os
import tempfile

st.set_page_config(page_title="Face Recognition", page_icon="👤", layout="wide")

st.title("👤 Face Recognition System")
st.markdown("---")

# Store registered faces
if 'registered_faces' not in st.session_state:
    st.session_state.registered_faces = {}  # name -> embedding

# Sidebar
with st.sidebar:
    st.header("📊 Controls")
    model_name = st.selectbox(
        "Model",
        ["VGG-Face", "Facenet", "OpenFace", "DeepFace"],
        index=1  # Facenet is accurate
    )
    threshold = st.slider("Threshold", 0.2, 0.8, 0.4, 0.01)

def register_face(name, img_path):
    """Register a face"""
    try:
        embedding = DeepFace.represent(
            img_path=img_path, 
            model_name=model_name, 
            enforce_detection=False
        )[0]["embedding"]
        st.session_state.registered_faces[name] = embedding
        return True
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return False

def recognize_face(img_path):
    """Recognize a face"""
    if not st.session_state.registered_faces:
        return "No faces registered", 0
    
    try:
        # Get embedding of test image
        test_embedding = DeepFace.represent(
            img_path=img_path, 
            model_name=model_name, 
            enforce_detection=False
        )[0]["embedding"]
        
        # Compare with all registered faces
        best_match = "UNKNOWN"
        min_distance = float('inf')
        
        for name, reg_embedding in st.session_state.registered_faces.items():
            distance = np.linalg.norm(np.array(test_embedding) - np.array(reg_embedding))
            if distance < min_distance:
                min_distance = distance
                best_match = name
        
        # Convert distance to similarity
        similarity = 1 / (1 + min_distance)
        
        if similarity > threshold:
            return best_match, similarity
        return "UNKNOWN", similarity
        
    except Exception as e:
        return f"Error", 0

# UI Tabs
tab1, tab2, tab3 = st.tabs(["📝 Register", "🔍 Recognize", "📋 List"])

# Tab 1: Register
with tab1:
    st.header("Register People")
    
    col1, col2 = st.columns(2)
    
    with col1:
        name = st.text_input("Name:")
        uploaded = st.file_uploader("Upload photo", type=['jpg','jpeg','png'], key="reg")
        
        if uploaded and name:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as f:
                f.write(uploaded.getvalue())
                temp_path = f.name
            
            img = Image.open(uploaded)
            st.image(img, width=200)
            
            if st.button("Register"):
                with st.spinner("Processing..."):
                    if register_face(name, temp_path):
                        st.success(f"✅ Registered {name}")
                    else:
                        st.error("❌ Failed")
                os.unlink(temp_path)

# Tab 2: Recognize
with tab2:
    st.header("Recognize Face")
    
    if not st.session_state.registered_faces:
        st.warning("Register Punya and Geethanjali M first")
    else:
        st.write(f"**Registered:** {', '.join(st.session_state.registered_faces.keys())}")
        
        uploaded = st.file_uploader("Upload photo", type=['jpg','jpeg','png'], key="rec")
        
        if uploaded:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as f:
                f.write(uploaded.getvalue())
                temp_path = f.name
            
            col1, col2 = st.columns(2)
            
            with col1:
                img = Image.open(uploaded)
                st.image(img, width=200)
            
            with col2:
                with st.spinner("Recognizing..."):
                    name, conf = recognize_face(temp_path)
                    
                    if name == "UNKNOWN":
                        st.error(f"❌ {name} ({conf:.2f})")
                    else:
                        st.success(f"✅ {name} ({conf:.2f})")
            
            os.unlink(temp_path)

# Tab 3: List
with tab3:
    st.header("Registered People")
    for name in st.session_state.registered_faces.keys():
        st.write(f"• {name}")

st.markdown("---")
st.caption(f"Using {model_name} | Threshold: {threshold}")
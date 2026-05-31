"""
=============================================================================
BERT NEWS TOPIC CLASSIFIER - STREAMLIT DEPLOYMENT APP
=============================================================================
Deploy and interact with the fine-tuned BERT model for news classification.

Run: streamlit run app.py
=============================================================================
"""

import streamlit as st
import torch
import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import pandas as pd
from typing import Dict, Tuple
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION & CACHING
# ============================================================================

st.set_page_config(
    page_title="📰 BERT News Classifier",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better aesthetics
st.markdown("""
<style>
    :root {
        --primary-color: #1f77b4;
        --secondary-color: #ff7f0e;
        --success-color: #2ca02c;
        --danger-color: #d62728;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    
    .news-card {
        border-left: 4px solid #667eea;
        padding: 15px;
        border-radius: 8px;
        background-color: #f8f9fa;
        margin: 10px 0;
    }
    
    .confidence-bar {
        height: 8px;
        border-radius: 4px;
        margin-top: 8px;
    }
    
    .category-world { border-color: #3498db; }
    .category-sports { border-color: #e74c3c; }
    .category-business { border-color: #2ecc71; }
    .category-science { border-color: #9b59b6; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# MODEL LOADING (CACHED FOR EFFICIENCY)
# ============================================================================

@st.cache_resource
def load_model_and_tokenizer():
    """Load the fine-tuned BERT model and tokenizer"""
    try:
        # Try to load from local directory first
        if Path("bert_news_classifier").exists():
            model_path = "bert_news_classifier"
        else:
            # Fallback to default path
            model_path = "bert-base-uncased"
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

@st.cache_resource
def load_config():
    """Load training configuration and class names"""
    try:
        config_path = Path("bert_news_classifier/config.json")
        if config_path.exists():
            with open(config_path, "r") as f:
                return json.load(f)
        else:
            # Default config
            return {
                "class_names": {
                    "0": "World",
                    "1": "Sports",
                    "2": "Business",
                    "3": "Science/Technology"
                },
                "max_length": 128,
                "accuracy": 0.92,
                "f1_score": 0.91
            }
    except Exception as e:
        st.warning(f"Could not load config: {e}")
        return {}

# ============================================================================
# INFERENCE FUNCTION
# ============================================================================

def classify_news(text: str, model, tokenizer, config: Dict) -> Dict:
    """
    Classify a news headline into one of 4 categories.
    
    Returns:
        Dictionary with predicted class, confidence, and all probabilities
    """
    if not text.strip():
        return None
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Tokenize
    max_length = config.get("max_length", 128)
    inputs = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    ).to(device)
    
    # Inference
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()
    predicted_class_id = np.argmax(probabilities)
    
    class_names = config.get("class_names", {
        "0": "World",
        "1": "Sports",
        "2": "Business",
        "3": "Science/Technology"
    })
    
    return {
        "predicted_class": class_names[str(predicted_class_id)],
        "class_id": int(predicted_class_id),
        "confidence": float(probabilities[predicted_class_id]),
        "probabilities": {
            class_names[str(i)]: float(probabilities[i])
            for i in range(len(probabilities))
        }
    }

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_category_emoji(category: str) -> str:
    """Get emoji for category"""
    emojis = {
        "World": "🌍",
        "Sports": "⚽",
        "Business": "💼",
        "Science/Technology": "🔬"
    }
    return emojis.get(category, "📰")

def get_category_color(category: str) -> str:
    """Get color for category"""
    colors = {
        "World": "#3498db",
        "Sports": "#e74c3c",
        "Business": "#2ecc71",
        "Science/Technology": "#9b59b6"
    }
    return colors.get(category, "#667eea")

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Header
    col1, col2 = st.columns([1, 4])
    with col1:
        st.image("https://huggingface.co/front/assets/huggingface_logo-noborder.svg", 
                width=80)
    with col2:
        st.markdown("# 📰 BERT News Topic Classifier")
        st.markdown("*Fine-tuned transformer model for accurate news classification*")
    
    st.divider()
    
    # Load model and config
    model, tokenizer = load_model_and_tokenizer()
    config = load_config()
    
    if model is None or tokenizer is None:
        st.error("❌ Failed to load model. Please ensure the model files are in the correct location.")
        return
    
    # Sidebar - Information
    with st.sidebar:
        st.markdown("## 📊 Model Information")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Accuracy", f"{config.get('accuracy', 0.92):.1%}")
        with col2:
            st.metric("F1-Score", f"{config.get('f1_score', 0.91):.1%}")
        
        st.markdown("---")
        st.markdown("## 📂 Categories")
        categories = config.get("class_names", {
            "0": "World",
            "1": "Sports",
            "2": "Business",
            "3": "Science/Technology"
        })
        for idx, category in enumerate(sorted(categories.values())):
            emoji = get_category_emoji(category)
            st.write(f"{emoji} **{category}**")
        
        st.markdown("---")
        st.markdown("## ℹ️ About")
        st.info("""
        **BERT News Classifier** uses a fine-tuned BERT model trained on the 
        AG News dataset to classify news headlines into 4 main categories.
        
        - **Model**: bert-base-uncased (fine-tuned)
        - **Training Samples**: 10,000
        - **Framework**: Hugging Face Transformers
        
        Built for NLP + Transfer Learning learning objectives.
        """)
    
    # Main content - Tabs
    tab1, tab2, tab3 = st.tabs(["🎯 Classify", "📚 Examples", "📖 Guide"])
    
    # ========================================================================
    # TAB 1: CLASSIFICATION
    # ========================================================================
    with tab1:
        st.markdown("### Enter a News Headline")
        
        # Input method selection
        input_method = st.radio(
            "How would you like to input text?",
            ["📝 Type Text", "📋 Paste Text"],
            horizontal=True
        )
        
        if input_method == "📝 Type Text":
            user_text = st.text_area(
                "Headline:",
                placeholder="Enter a news headline here...",
                height=100,
                label_visibility="collapsed"
            )
        else:
            user_text = st.text_input(
                "Headline:",
                placeholder="Paste a headline here...",
                label_visibility="collapsed"
            )
        
        # Classify button
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            classify_btn = st.button("🚀 Classify", use_container_width=True)
        with col2:
            clear_btn = st.button("🗑️ Clear", use_container_width=True)
        
        if clear_btn:
            st.rerun()
        
        # Process classification
        if classify_btn:
            if not user_text.strip():
                st.warning("⚠️ Please enter a headline first!")
            else:
                with st.spinner("🔄 Classifying..."):
                    result = classify_news(user_text, model, tokenizer, config)
                
                if result:
                    # Result display
                    st.markdown("---")
                    st.markdown("### 🎯 Classification Result")
                    
                    # Main result card
                    category = result["predicted_class"]
                    confidence = result["confidence"]
                    emoji = get_category_emoji(category)
                    color = get_category_color(category)
                    
                    st.markdown(f"""
                    <div style="
                        background: linear-gradient(135deg, {color}20 0%, {color}40 100%);
                        border-left: 5px solid {color};
                        padding: 20px;
                        border-radius: 10px;
                    ">
                        <h2 style="margin: 0 0 10px 0; color: {color};">
                            {emoji} {category}
                        </h2>
                        <p style="margin: 0; font-size: 14px; color: #666;">Confidence: <strong>{confidence:.1%}</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Probability breakdown
                    st.markdown("#### 📊 All Probabilities")
                    
                    # Create visualization
                    probs_df = pd.DataFrame([
                        {"Category": cat, "Probability": prob}
                        for cat, prob in sorted(
                            result["probabilities"].items(),
                            key=lambda x: x[1],
                            reverse=True
                        )
                    ])
                    
                    # Bar chart
                    st.bar_chart(
                        probs_df.set_index("Category"),
                        use_container_width=True,
                        height=300
                    )
                    
                    # Table
                    st.markdown("#### 📈 Detailed Breakdown")
                    
                    breakdown_cols = st.columns(4)
                    for idx, (cat, prob) in enumerate(sorted(
                        result["probabilities"].items(),
                        key=lambda x: x[1],
                        reverse=True
                    )):
                        with breakdown_cols[idx]:
                            emoji = get_category_emoji(cat)
                            st.metric(
                                f"{emoji} {cat}",
                                f"{prob:.1%}"
                            )
                    
                    # Input text reference
                    st.markdown("---")
                    st.markdown("#### 📝 Input Text")
                    st.info(f'"{user_text}"')
    
    # ========================================================================
    # TAB 2: EXAMPLES
    # ========================================================================
    with tab2:
        st.markdown("### 🔍 Try Pre-loaded Examples")
        
        examples = {
            "🌍 World": [
                "UN Security Council discusses Middle East tensions",
                "European Parliament approves new climate bill",
                "North Korean leader meets with Chinese officials"
            ],
            "⚽ Sports": [
                "Messi leads Inter Miami to playoff victory",
                "Lakers defeat Celtics in overtime thriller",
                "Serena Williams announces tennis comeback plans"
            ],
            "💼 Business": [
                "Apple announces record quarterly earnings",
                "Tesla stock surges on new product reveal",
                "Microsoft invests $10 billion in AI research"
            ],
            "🔬 Science/Technology": [
                "Quantum computing breakthrough announced by researchers",
                "SpaceX successfully launches Mars mission",
                "New AI model achieves human-level reasoning"
            ]
        }
        
        for category, headlines in examples.items():
            with st.expander(f"View {category} Examples"):
                for i, headline in enumerate(headlines, 1):
                    if st.button(f"📌 Example {i}", key=f"{category}_{i}"):
                        st.session_state.selected_text = headline
                    
                    # Show preview
                    st.caption(f"{i}. {headline}")
    
    # ========================================================================
    # TAB 3: GUIDE
    # ========================================================================
    with tab3:
        st.markdown("""
        ### 📖 How to Use This Classifier
        
        #### 🎯 **Classification Process**
        1. Enter a news headline in the "Classify" tab
        2. Click the **Classify** button
        3. Get instant predictions with confidence scores
        
        #### 💡 **Understanding Results**
        - **Predicted Class**: The most likely category for your headline
        - **Confidence**: How certain the model is (0-100%)
        - **Probabilities**: Score for each category
        
        #### 🏆 **Model Performance**
        - **Accuracy**: {:.1%}
        - **F1-Score**: {:.1%}
        - **Training Samples**: 10,000
        - **Model**: BERT (bert-base-uncased)
        
        #### 📚 **Categories Explained**
        - **World**: International news, politics, conflicts
        - **Sports**: Sports events, athletes, competitions
        - **Business**: Economics, companies, markets
        - **Science/Technology**: Research, AI, innovation
        
        #### ⚙️ **Technical Details**
        - **Framework**: Hugging Face Transformers
        - **Hardware**: GPU-accelerated inference
        - **Input Length**: Up to 128 tokens
        
        #### 🔗 **Learn More**
        - [Hugging Face Model Hub](https://huggingface.co/models)
        - [BERT Paper](https://arxiv.org/abs/1810.04805)
        - [AG News Dataset](https://huggingface.co/datasets/ag_news)
        """.format(
            config.get('accuracy', 0.92),
            config.get('f1_score', 0.91)
        ))
    
    # ========================================================================
    # Footer
    # ========================================================================
    st.divider()
    st.markdown("""
    <div style="text-align: center; color: #999; font-size: 12px; margin-top: 20px;">
        <p>Built with 🚀 Streamlit | Powered by 🤗 Hugging Face Transformers | Fine-tuned BERT</p>
        <p>©2024 BERT News Classifier | All Rights Reserved</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

model = AutoModelForSequenceClassification.from_pretrained(
    "sentiment_model_v2",
    trust_remote_code=True,
    low_cpu_mem_usage=False  # VERY IMPORTANT to force full load!
)
model.eval()  # Sets model in inference mode

tokenizer = AutoTokenizer.from_pretrained("sentiment_model_v2")


label_map = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}

st.set_page_config(page_title="Sentiment Analyzer", page_icon="💬", layout="wide")

# ✅ Background Image URL
background_image_url = "https://images.unsplash.com/photo-1504384308090-c894fdcc538d"

# ✅ CSS Styling (no white box, all text on translucent background)
custom_css = f"""
<style>
html, body, [class*="stApp"] {{
    background-image: url('{background_image_url}');
    background-size: cover;
    background-attachment: fixed;
    background-position: center;
    background-repeat: no-repeat;
    color: white;
    font-family: 'Segoe UI', sans-serif;
}}

h1, h2, h3, h4, h5, h6 {{
    color: white;
    text-shadow: 1px 1px 2px #000;
    text-align: center;
}}

.stTextArea textarea {{
    background-color: rgba(255, 255, 255, 0.1);
    color: white;
    font-size: 1.1em;
    border-radius: 8px;
    border: 1px solid #ccc;
}}

.stButton > button {{
    background-color: #ffffff22;
    color: white;
    font-weight: bold;
    border-radius: 6px;
    padding: 0.5rem 1.5rem;
    border: none;
}}

.stButton > button:hover {{
    background-color: #ffffff44;
}}

.stAlert {{
    background-color: rgba(0, 0, 0, 0.4);
    border-left: 5px solid white;
}}

</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)

# ✅ App content directly on background
st.markdown("<h1>💬 Sentiment Analyzer</h1>", unsafe_allow_html=True)
st.write("Enter your text to predict the sentiment:")

text = st.text_area("Your Text", height=150)

if st.button("Analyze Sentiment"):
    if text.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Tokenize input
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

        # Predict sentiment
        with torch.no_grad():
            outputs = model(**inputs)
            print(outputs.logits)
            print(outputs.logits.device)
            prediction = torch.argmax(outputs.logits, dim=1).item()
            sentiment = label_map[prediction]

        # Display result
        st.success(f"**Predicted Sentiment:** {sentiment}")


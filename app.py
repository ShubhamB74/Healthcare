import time
import streamlit as st
from transformers import pipeline
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from textblob import TextBlob

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# Use a pipeline as a high-level helper
chatbot = pipeline("text-generation", model="ContactDoctor/Bio-Medical-Llama-3-2-1B-CoT-012025")

stop_words = set(stopwords.words('english'))

# Simple function to clean and preprocess user input
def clean_input(user_input):
    word_tokens = word_tokenize(user_input)
    filtered_input = [word for word in word_tokens if word.lower() not in stop_words and word.isalnum()]
    return " ".join(filtered_input)

# Function for sentiment analysis to adjust chatbot tone
def analyze_sentiment(user_input):
    sentiment = TextBlob(user_input).sentiment
    return sentiment.polarity  # Range: -1 (negative) to 1 (positive)

def healthcare_chatbot(user_input):
    cleaned_input = clean_input(user_input)
    sentiment_score = analyze_sentiment(user_input)

    # Handle specific cases based on sentiment
    if sentiment_score < -0.5:
        return "I'm sorry you're feeling upset. Please let me know how I can help you, and I will try to assist you the best I can."
    
    # Simple rule-based responses for common queries
    if "symptom" in cleaned_input:
        return "It seems like you're experiencing symptoms. Please consult a doctor for accurate advice."
    elif "appointment" in cleaned_input:
        return "Would you like me to schedule an appointment with a doctor?"
    elif "medication" in cleaned_input:
        return "It's important to take your prescribed medications regularly. If you have concerns, consult your doctor."
    else:
        # Generate more personalized responses
        response = chatbot(user_input, max_length=300, num_return_sequences=1)
        return response[0]['generated_text']


# Streamlit web app interface
def main():
    st.set_page_config(page_title="Healthcare Assistant Chatbot", page_icon="💬")
    st.title("Healthcare Assistant Chatbot")
    
    with st.sidebar:
        st.header("Chatbot Navigation")
        st.write("You can ask about symptoms, appointments, medication, or other healthcare-related queries.")
        st.markdown("### About")
        st.write("This is a healthcare assistant chatbot built to help you with healthcare-related queries.")
    
    st.markdown("""
        <style>
            .stTextInput>div>div>input {
                height: 80px;
                font-size: 16px;
                padding: 15px;
            }
            .stButton>button {
                background-color: #4CAF50;
                color: white;
                font-size: 16px;
                border-radius: 5px;
                padding: 10px 20px;
            }
            .stMarkdown {
                font-size: 16px;
                font-weight: 400;
                line-height: 1.5;
            }
            .stWrite {
                font-size: 18px;
                font-weight: 500;
            }
        </style>
    """, unsafe_allow_html=True)
    
    user_input = st.text_area("How can I assist you today?", "", height=120)
  
    if st.button("Submit"):
        if user_input:
            with st.spinner("Thinking..."):
                time.sleep(2)  # Simulating a slight delay for better UX
                st.write("**User:** ", user_input)
                response = healthcare_chatbot(user_input)
                st.markdown(f"**Healthcare Assistant:**\n\n{response}")
        else:
            st.warning("Please enter a query.")

if __name__ == "__main__":
    main()

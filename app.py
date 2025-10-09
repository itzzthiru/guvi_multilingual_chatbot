import streamlit as st
from chatbot import Chatbot

st.set_page_config(page_title="GUVI Multilingual Chatbot", page_icon="🤖", layout="centered")

@st.cache_resource(show_spinner=False)
def get_bot():
    return Chatbot()

def main():
    st.title("🤖 GUVI Multilingual Chatbot")
    st.caption("Type in English or your native language. I'll translate, answer, and translate back.")

    user_input = st.text_input("Your question:")
    if st.button("Ask") and user_input.strip():
        bot = get_bot()
        with st.spinner("Thinking..."):
            try:
                response_data = bot.get_response(user_input.strip())
                
                # Display FAQ answers
                if response_data["faq_answers"]:
                    st.markdown(f"**FAQ Answer:** {response_data['faq_answers'][0][0]}")
                
                # Display GUVI content
                if response_data["guvi_paragraphs"]:
                    st.markdown(f"**GUVI Content:** {response_data['guvi_paragraphs'][0][0]}")
                    
                # If no results found
                if not response_data["faq_answers"] and not response_data["guvi_paragraphs"]:
                    st.markdown("Sorry, I couldn't find an answer to your question.")
                    
            except Exception as e:
                st.error(f"Oops, something went wrong: {e}")

if __name__ == "__main__":
    main()
# app.py
import streamlit as st
from chatbot import Chatbot
import time

# ---------------------------- #
# Streamlit page setup
# ---------------------------- #
st.set_page_config(
    page_title="GUVI Multilingual Chatbot — Project",
    page_icon="🤖",
    layout="centered"
)

# ---------------------------- #
# Cache chatbot instance (to avoid reloading model)
# ---------------------------- #
@st.cache_resource(show_spinner=False)
def get_bot():
    return Chatbot()

# ---------------------------- #
# Initialize session state
# ---------------------------- #
def init_session():
    if "history" not in st.session_state:
        st.session_state.history = []  # [(speaker, text)]
    if "last_lang" not in st.session_state:
        st.session_state.last_lang = "eng_Latn"
    if "last_results" not in st.session_state:
        st.session_state.last_results = None

# ---------------------------- #
# Display conversation history
# ---------------------------- #
def render_chat():
    if not st.session_state.history:
        st.info("No conversation yet. Ask your first question below 👇")
    else:
        for speaker, text in st.session_state.history:
            if speaker == "user":
                st.markdown(f"🧑 **You:** {text}")
            else:
                st.markdown(f"🤖 **Bot:** {text}")

# ---------------------------- #
# Main function
# ---------------------------- #
def main():
    init_session()
    st.title("🤖 GUVI Multilingual Chatbot — Project")
    st.caption("Ask me about GUVI, CodeKata, WebKata, or any course — in your own language.")
    st.markdown("---")

    bot = get_bot()

    # Input field and detected language
    cols = st.columns([3, 1])
    with cols[0]:
        user_input = st.chat_input("💬 Type your question here...")
    with cols[1]:
        st.markdown(f"**Detected:** {st.session_state.last_lang}")

    if user_input:
        user_input = user_input.strip()
        st.session_state.history.append(("user", user_input))

        # Run bot response
        with st.spinner("Processing..."):
            try:
                start_time = time.time()
                response_data = bot.get_response(user_input, top_k=3)
                response_time = time.time() - start_time

                st.session_state.last_lang = response_data.get("detected_lang_code", "eng_Latn")

                faqs = response_data.get("faq_answers", [])
                guvi = response_data.get("guvi_paragraphs", [])
                gen = response_data.get("generative_answers", [])

                # Primary text for conversation
                primary_text = None
                if gen:
                    primary_text = gen[0][0]
                elif faqs:
                    primary_text = faqs[0][0]
                elif guvi:
                    primary_text = guvi[0][0]
                else:
                    primary_text = "I couldn’t find an answer. Try rephrasing your question."

                st.session_state.history.append(("bot", primary_text))

                # Save detailed results to display later
                st.session_state.last_results = {
                    "faqs": faqs,
                    "guvi": guvi,
                    "gen": gen,
                    "latency": response_time
                }

            except Exception as e:
                st.error(f"⚠️ Error: {e}")
                st.session_state.history.append(("bot", "Sorry, something went wrong."))
                st.session_state.last_results = None

    # ---------------------------- #
    # Display conversation first
    # ---------------------------- #
    st.markdown("---")
    st.header("💬 Conversation")
    render_chat()

    # ---------------------------- #
    # Show detailed results (below)
    # ---------------------------- #
    if st.session_state.last_results:
        results = st.session_state.last_results
        faqs = results["faqs"]
        guvi = results["guvi"]
        gen = results["gen"]
        latency = results["latency"]

        st.markdown("---")
        st.header("📊 Detailed Results")

        if faqs:
            st.markdown("✅ **FAQ Matches:**")
            for text, score, *rest in faqs:
                st.markdown(f"- {text} _(score: {score:.3f})_")

        if guvi:
            st.markdown("📘 **GUVI Paragraph Matches:**")
            for text, score, *rest in guvi:
                st.markdown(f"- {text} _(score: {score:.3f})_")

        if gen:
            st.markdown("🧠 **Generative Answers (Fallback):**")
            for text, score, *rest in gen:
                st.markdown(f"- {text}")

        st.caption(f"⏱️ Response time: {latency:.2f}s")

# ---------------------------- #
# Run the app
# ---------------------------- #
if __name__ == "__main__":
    main()

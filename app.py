import os
import validators 
import streamlit as st
#from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.schema import Document
from urllib.parse import urlparse, parse_qs
from langchain_core.prompts import PromptTemplate
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import UnstructuredURLLoader

# Load environment variables
#load_dotenv(r'F:\Udemy\langchain\.env')

# Streamlit app configuration
st.set_page_config(page_title="Langchain: Summarize text from YT or website")
st.title("Summarize text from YT or website")
st.subheader("Summarize URL")

# Sidebar for model selection
model_options = [
    "llama-3.1-70b-versatile",
    "Gemma-7b-It",
    "Gemma2-9b-It",
    "mixtral-8x7b-32768",
    "llama3-70b-8192",
    "llama3-groq-70b-8192-tool-use-preview",
]
selected_model = st.sidebar.selectbox("Select the Groq Model", model_options)
language = st.sidebar.text_input(
    "Preferred Language",
    placeholder="e.g., English, Spanish, French",
    label_visibility="visible"
)
# Default to English if no input is provided
language = language if language else "English"

# LLM setup
groq_api_key=os.getenv("GROQ_API_KEY")
llm = ChatGroq(groq_api_key=groq_api_key, model=selected_model)

def get_video_id(url):
    """
    Extracts the video_id from a YouTube URL.
    """
    parsed_url = urlparse(url)
    if parsed_url.hostname in ['www.youtube.com', 'youtube.com']:
        query_params = parse_qs(parsed_url.query)
        return query_params.get('v', [None])[0]
    elif parsed_url.hostname == 'youtu.be':
        return parsed_url.path.lstrip('/')
    return None

prompt_template = """
Summarize the following content into a maximum of 500 words, ensuring the summary is clear, concise, 
and retains the most important information. 
The summary should be written in the specified language: {language}.
Content: {text}
"""

prompt = PromptTemplate(template=prompt_template,input_variables=['text'])

generic_url = st.text_input("URL",label_visibility='collapsed',placeholder="Enter the URL to Summarize")

if st.button("Summarize the content from YT or website"):
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please provide the information to get started")
    elif not validators.url(generic_url):
        st.error("Please enter a valid URL. It can be a YouTube video URL or a website URL")
    else:
        try:
            with st.spinner("Waiting..."):
                try:
                    if "youtube.com" in generic_url or "youtu.be" in generic_url:
                        video_id = get_video_id(generic_url)
                        if video_id:
                            try:
                                transcript = YouTubeTranscriptApi.get_transcript(video_id)
                                text_content = " ".join([t['text'] for t in transcript])
                                docs = [Document(page_content=text_content)]  # Wrap in a Document-like object
                            except Exception as e:
                                st.error(f"Error fetching transcript: {e}")
                                st.stop()
                        else:
                            st.error("Invalid YouTube URL")
                            st.stop()
                    else:
                        loader = UnstructuredURLLoader(
                            urls=[generic_url],
                            ssl_verify=False,
                            headers={
                                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"
                            }
                        )
                        docs = loader.load()  # This should already return document-like objects
                except Exception as e:
                    st.error(f"Failed to load content from the URL: {str(e)}")
                    st.stop()

                # Summarization chain
                chain = load_summarize_chain(llm, chain_type='stuff', prompt=prompt)
                output_summary = chain.invoke({"input_documents":docs,"language":language})
                summary_text = output_summary['output_text']

                # Store the summary in session state
                st.session_state['summary'] = summary_text
                st.success("Summary generated successfully!")
                st.write(summary_text)

                #st.success(output_summary['output_text'])

        except Exception as e:
            st.exception(f"Exception: {e}")

# Question answering section
if 'summary' in st.session_state:
    st.subheader("Ask questions about the summary")
    user_question = st.text_input("Your question", key="user_question")

    # Initialize conversation history in session state
    if 'conversation' not in st.session_state:
        st.session_state['conversation'] = []

    if st.button("Ask"):
        if user_question.strip():
            try:
                # Define a custom prompt
                custom_prompt = (
                    f"You are an expert assistant. Based on the summary provided below, "
                    f"answer the question in a detailed manner(you can add more to make it complete):\n\n"
                    f"Summary:\n{st.session_state['summary']}\n\n"
                    f"Question:\n{user_question}"
                )

                # Get response from ChatGroq
                response = llm.invoke(custom_prompt)
                
                # Extract the answer from the response
                # ChatGroq returns a message object, so we need to get the content
                answer = response.content

                # Save the question and answer
                st.session_state['conversation'].append({
                    "question": user_question,
                    "answer": answer
                })

                # Display the answer
                st.write(f"**Answer:** {answer}")

            except Exception as e:
                st.error(f"Error processing the question: {e}")
        else:
            st.error("Please enter a question.")

    # Display conversation history
    if st.session_state['conversation']:
        st.subheader("Conversation History")
        for idx, qa in enumerate(st.session_state['conversation']):
            st.write(f"**Q{idx+1}:** {qa['question']}")
            st.write(f"**A{idx+1}:** {qa['answer']}")
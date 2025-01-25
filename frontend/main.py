import time
import streamlit as st
import requests
from Utils.conversation_memory import ConversationMemory
from Utils.Sidebar import Sidebar

def create_page():
    # Initialize conversation_id in session state if not present
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = None
        
    if "llm_model" not in st.session_state : 
        st.session_state["llm_model"] = {"provider": "ollama", "model_name": "llama3.1"}       

    col1, _, _ = st.columns([2, 2,2])  # Adjust the ratio as needed
    with col1:
        if st.button('Clear Chat History', use_container_width= True, type = "primary") :
            st.session_state.memory.clear()
            st.rerun()


    # Initialize messages if not present
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationMemory()    

    # Display chat messages
    for message in st.session_state.memory.history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    if prompt := st.chat_input(disabled=False):
        st.session_state.memory.add_exchange({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer, source_documents = generate_response(prompt)
                    # if source_documents:
                    #     answer += "\nsources:"
                    #     for doc in source_documents:
                    #         answer += f"\n- {doc}"
                    st.session_state.memory.add_exchange({"role": "assistant", "content": answer})
                    st.markdown(answer)
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
    return True
    
def generate_response(prompt_input):
    conversation_history = st.session_state.memory.get_context()
    payload = {
        "user_input": prompt_input,
        "conversation_history": conversation_history,
        "llm_model": st.session_state["llm_model"]
               }
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.post("http://127.0.0.1:8000/chat", json=payload, headers=headers)
        response.raise_for_status()
        output = response.json()
        return output["answer"], output["source_documents"]

    except requests.RequestException as e:
        raise Exception(f"API request failed: {str(e)}")

if __name__ == "__main__":
    Sidebar()
    create_page()
    

import time
import streamlit as st

class Sidebar:
    def __init__(self):
        self.render_sidebar()

    def render_sidebar(self):        
        with st.sidebar:
            st.title("Settings")            
            llm_model_choice = st.selectbox(
                "Choose LLM Provider",
                options=["Ollama","OpenAI", "Hugging Face", "Groq", "Together.AI"],
                index=0
            )
                
            if llm_model_choice == "Ollama":
                provider = "ollama" 
                ollama_model_name = st.text_input("Enter Ollama Model Name", value = 'llama3.1')
                    
            elif llm_model_choice == "OpenAI":
                provider = "openai"
                openai_model_name = st.text_input("Enter OpenAI Model Name")
                openai_api_key = st.text_input("Enter your API Token", type = "password")
                
            elif llm_model_choice == "Hugging Face":
                provider = "huggingface"
                huggingface_model_name_id = st.text_input("Enter Hugging Face Model/Repo Name ID")
                huggingface_api_token = st.text_input("Enter your API Token", type = "password")
                    
            elif llm_model_choice == "Groq":
                provider = "groq"
                groq_api_key = st.text_input("Enter Groq API key", type="password")
                model = st.text_input("Enter model name")
                    
            elif llm_model_choice == "Together.AI":
                provider = "together"
                together_api_key = st.text_input("Enter Together AI API key", type="password")
                model = st.text_input("Enter model name")

            if st.button("Save Model Choices", type= 'primary'):
                if provider == "ollama":
                    st.session_state["llm_model"] = {"provider": "ollama", "model_name": ollama_model_name.strip()}
                        
                elif provider == "openai":
                    st.session_state["llm_model"] = {"provider": "openai", "model_name": openai_model_name.strip(), "api_key": openai_api_key }
                        
                elif provider == "huggingface":
                    st.session_state["llm_model"] = {"provider": "huggingface","api_key": huggingface_api_token, "model_id": huggingface_model_name_id.strip()}
                        
                elif provider == "groq":
                    st.session_state["llm_model"] = {"provider": "groq", "api_key": groq_api_key.strip(),"model_name": model.strip()}
                        
                elif provider == "together":
                    st.session_state["llm_model"] = {"provider": "together", "api_key": together_api_key.strip(),"model": model.strip()}

                st.success("Model choices saved successfully.")
                time.sleep(0.8)
                st.rerun()
        
                return True
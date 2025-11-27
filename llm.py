
from llama_cpp import Llama

class SLM_TALWAR:
    def __init__(self):
        # print(f"Loading LLM from: {config.LLM_MODEL_PATH}")

        # self.llm = Llama(
        #     model_path=config.LLM_MODEL_PATH, 
        #     n_ctx=4096,
        #     n_gpu_layers=-1, # -1 to offload all layers to GPU
        #     verbose=True
        # )
        print("LLM loaded successfully.")
        self.model = Llama(
            model_path="model/LLM/Incresed4_Model_q4.gguf",
            n_ctx=4056,
            n_threads=8
        )
        self.system_message = "You are TALWAR, a friendly and knowledgeable AI assistant developed by DGIS.When the user explicitly asks your name or who developed/created you, you must reply exactly:'TALWAR, developed by DGIS.' Do not give this response unless the user directly asks."
        self.rag_message = """You are Talwar, a helpful AI assistant specialized in explaining document content. 

Your task is to answer questions based on the provided context. Follow these rules:
1. Answer the question using information from the context
2. Be comprehensive and explain concepts clearly
3. If the context contains relevant information, use it to formulate your answer
4. Only say you don't know if the context has absolutely no relevant information
5. You can synthesize and explain information from the context in your own words
"""

    def generate(self, query):
        """Generate response without conversation history (legacy method)"""
        messages = [
            {
                "role": "system",
                "content": self.system_message
            },
            {
                "role": "user",
                "content": query
            }
        ]
        
        response = self.model.create_chat_completion(
            messages=messages,
            max_tokens=4096,
            temperature=0.7
        )
        
        assistant_reply = response['choices'][0]['message']['content']
        return assistant_reply
    
    def generate_with_history(self, query, conversation_history):
        """
        Generate response with a sliding window of conversation history.
        """
        # Build messages array with system message
        messages = [
            {
                "role": "system",
                "content": self.system_message
            }
        ]
        
        # --- CHANGED: SIMPLIFIED SLIDING WINDOW ---
        # Define the max number of *total* messages (user + assistant) to keep in the window
        # 10 messages = 5 turns
        MAX_MESSAGES_IN_WINDOW = 10
            
        # Get the full history *including* the current user query
        # and slice it to be at most MAX_MESSAGES_IN_WINDOW items
        windowed_history = conversation_history[-MAX_MESSAGES_IN_WINDOW:]
            
        # Add this window to our messages list
        messages.extend(windowed_history)
        
        try:
            response = self.model.create_chat_completion(
                messages=messages,
                max_tokens=4096,
                temperature=0.7
            )
            
            assistant_reply = response['choices'][0]['message']['content']
            return assistant_reply
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return "I apologize, but I encountered an error generating a response. Please try again."

    def rag_chat(self, query, relevant_chunks):
        """Generate response using RAG approach with relevant document chunks"""
        # Build context from relevant chunks
        context = "\n\n".join([chunk['payload']['text'] for chunk in relevant_chunks])
        
        print(context)
        messages = [
            {
                "role": "system",
                "content": self.rag_message
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}"
            }
        ]
        
        try:
            response = self.model.create_chat_completion(
                messages=messages,
                max_tokens=4096,
                temperature=0.7
            )
            
            assistant_reply = response['choices'][0]['message']['content']
            return assistant_reply
        
        except Exception as e:
            print(f"Error generating RAG response: {str(e)}")
            return "I apologize, but I encountered an error generating a response with the provided context. Please try again."
    
    def generate_rag_with_history(self, query, conversation_history, relevant_chunks):
        """
        Generate RAG response.
        NOTE: Per your request, conversation history is no longer used here to fix the role error.
        """
        # Build context from relevant chunks
        
        # --- FIX #2 HERE ---
        # Changed chunk.page_content to chunk['page_content']
        context = "\n\n".join([chunk['payload']['text'] for chunk in relevant_chunks])
        
        print("-----------------------------------------context")
        print(context)
        # Build messages array with system message
        messages = [
            {
                "role": "system",
                "content": self.rag_message
            }
        ]
        
        # --- CHANGED: REMOVED HISTORY LOOP AS REQUESTED ---
        # The 'conversation_history' parameter is now unused in this method.
        # This fixes the "Conversation roles must alternate" error.
        
        # Add current user query with context
        messages.append({
            "role": "user",
            "content": f"Context: \n{context} \n\n Question: {query}"
        })
        print('-'*100)
        print(messages)
        print('-'*100)
        try:
            print("Generating RAG response (stateless)...")
            response = self.model.create_chat_completion(
                messages=messages,
                max_tokens= 4096,
                temperature=0.7
            )
            
            assistant_reply = response['choices'][0]['message']['content']
            return assistant_reply
        
        except Exception as e:  
            print(f"Error generating RAG response: {str(e)}")
            return "I apologize, but I encountered an error generating a response with the provided context. Please try again."

    # --- NEW METHOD FOR SUMMARIZATION ---
    def summarize_conversation(self, conversation_history):
        """
        Generates a concise summary of the provided conversation history.
        """
        
        # 1. Format the conversation history into a single string
        history_str = ""
        for msg in conversation_history:
            role = "User" if msg['role'] == 'user' else "Assistant"
            history_str += f"{role}: {msg['content']}\n"
        
        if not history_str:
            return "[No conversation to summarize]"
            
        # 2. Create the messages payload for summarization
        # Using a dedicated system prompt for this task
        summary_system_prompt = "You are a helpful assistant. Your task is to provide a concise summary of the following conversation. Focus on the main topics and key information exchanged."
        
        messages = [
            {
                "role": "system",
                "content": summary_system_prompt
            },
            {
                "role": "user",
                "content": f"Please summarize this conversation:\n\n{history_str}"
            }
        ]
        
        # 3. Call the model
        try:
            response = self.model.create_chat_completion(
                messages=messages,
                max_tokens=256,  # Summaries should be concise
                temperature=0.3  # Lower temp for more factual summary
            )
            summary = response['choices'][0]['message']['content']
            return summary.strip()
        except Exception as e:
            print(f"Error generating summary: {str(e)}")
            return "[Error creating summary]"


# --- UPDATED MAIN CHAT LOOP ---
if __name__ == "__main__":
    slm = SLM_TALWAR()
    print("SLM(TALWAR): Hello! I'm SLM(TALWAR). How can I help you?")
    
    conversation_history = []
    
    # We will summarize every 3 turns. 
    # A "turn" is one user message and one assistant response.
    SUMMARY_INTERVAL = 3  # You can change this to 2
    
    while True:
        user_input = input(">>> ")
        if user_input.lower() in ['exit', 'quit']:
            break
        
        # Add user message to history
        conversation_history.append({
            'role': 'user',
            'content': user_input
        })
        
        # Generate response with history
        response = slm.generate_with_history(user_input, conversation_history)
        
        # Add assistant response to history
        conversation_history.append({
            'role': 'assistant',
            'content': response
        })
        
        print(f"SLM(TALWAR): {response}")

        # --- NEW SUMMARIZATION LOGIC ---
        
        # Calculate the number of full turns (user + assistant)
        num_turns = len(conversation_history) // 2
        
        # Check if the number of turns is a multiple of our interval
        if num_turns > 0 and num_turns % SUMMARY_INTERVAL == 0:
            print("\n[Generating conversation summary...]")
            
            # Pass the *current* history to the new summary method
            summary = slm.summarize_conversation(conversation_history)
            
            print(f"\n[CURRENT SUMMARY]: {summary}\n")
            
            # Optional: If you want to save tokens, you could now "prune"
            # the history by replacing it with the summary.
            # For example:
            # conversation_history = [
            #     {'role': 'system', 'content': f'Summary of previous chat: {summary}'}
            # ]
            # Be careful with this, as it will affect the next `generate_with_history` call.
            # For now, we just print the summary and keep the full history.
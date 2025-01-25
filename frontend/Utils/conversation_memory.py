class ConversationMemory:
    def __init__(self, max_turns=10):
        self.history = [{"role": "assistant", "content": "How may I assist you today?"}]
        self.max_turns = max_turns

    def add_exchange(self, message):
        self.history.append(message)

    def get_context(self):
        if len(self.history) > 8:
            return self.format_chat_history(self.history[-9:-1])
        else:
            return self.format_chat_history(self.history[:-1])
        
    def clear(self):
        self.history = [{"role": "assistant", "content": "How may I assist you today?"}]
        
    def format_chat_history(self, conversation_history):
        chat_history = []
        for entry in conversation_history:
            if entry['role'] == 'user':
                chat_history.append(("human", entry['content']))
            elif entry['role'] == 'assistant':
                chat_history.append(("ai", entry['content']))
        return chat_history

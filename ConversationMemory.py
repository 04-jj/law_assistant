from datetime import datetime
from typing import List


class ConversationMemory:
    """对话记忆管理类"""

    def __init__(self, max_history_turns: int = 5):
        self.max_history_turns = max_history_turns
        self.conversations = {}

    def add_message(self, conversation_id: str, role: str, content: str):
        """添加消息到对话历史"""
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = {
                'history': [],
                'created_at': datetime.now()
            }

        conversation = self.conversations[conversation_id]
        conversation['history'].append({
            'role': role,
            'content': content,
            'timestamp': datetime.now()
        })

        # 保留最近5轮对话
        max_messages = self.max_history_turns * 2
        if len(conversation['history']) > max_messages:
            conversation['history'] = conversation['history'][-max_messages:]

    def get_recent_history(self, conversation_id: str) -> List[dict]:
        """获取最近的对话历史"""
        if conversation_id not in self.conversations:
            return []

        return self.conversations[conversation_id]['history']

    def get_formatted_history(self, conversation_id: str) -> str:
        """获取格式化的对话历史"""
        history = self.get_recent_history(conversation_id)
        if not history:
            return "无对话历史"

        formatted = "最近的对话历史：\n"
        for i, msg in enumerate(history):
            speaker = "用户" if msg['role'] == 'user' else "助手"
            formatted += f"{i + 1}. {speaker}: {msg['content']}\n"

        return formatted

    def clear_conversation(self, conversation_id: str):
        """清空特定对话的记忆"""
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]

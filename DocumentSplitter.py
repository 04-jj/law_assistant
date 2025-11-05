from langchain_text_splitters import TextSplitter, RecursiveCharacterTextSplitter
from typing import List
import re


class DocumentSplitter(TextSplitter):
    """文档分块，按条款分割"""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def split_text(self, text: str) -> List[str]:
        """按法律条款分割文本"""

        # 匹配法律条款的正则表达式
        # 匹配 "第X条"、"第X条规定"、"《法律名称》第X条" 等格式
        article_pattern = r'第[零一二三四五六七八九十百千万\d]+条'

        # 找到所有条款的位置
        articles = []
        for match in re.finditer(article_pattern, text):
            articles.append({
                'start': match.start(),
                'text': match.group(),
                'content': ''
            })

        # 如果没有找到条款，回退到普通分块
        if not articles:
            return self._fallback_split(text)

        # 为每个条款提取内容
        chunks = []
        for i, article in enumerate(articles):
            start_pos = article['start']

            # 确定当前条款的结束位置（下一个条款的开始或文本结尾）
            if i < len(articles) - 1:
                end_pos = articles[i + 1]['start']
            else:
                end_pos = len(text)

            # 提取条款内容
            article_content = text[start_pos:end_pos].strip()

            # 如果条款内容过长，再进行细分
            if len(article_content) > self._chunk_size:
                sub_chunks = self._split_long_article(article_content)
                chunks.extend(sub_chunks)
            else:
                chunks.append(article_content)

        return chunks

    def _split_long_article(self, article_content: str) -> List[str]:
        """对过长的条款进行进一步分割"""
        # 按句号、分号等标点分割
        sentences = re.split(r'[。；;]', article_content)

        chunks = []
        current_chunk = ""

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # 如果当前块加上新句子不会超过限制
            if len(current_chunk) + len(sentence) + 1 <= self._chunk_size:
                if current_chunk:
                    current_chunk += "。" + sentence
                else:
                    current_chunk = sentence
            else:
                # 保存当前块并开始新块
                if current_chunk:
                    chunks.append(current_chunk + "。")
                    current_chunk = sentence
                else:
                    # 单个句子就超过限制，直接作为一个块
                    chunks.append(sentence + "。")
                    current_chunk = ""

        # 添加最后一个块
        if current_chunk:
            chunks.append(current_chunk + "。")

        return chunks

    def _fallback_split(self, text: str) -> List[str]:
        """回退到普通分块策略"""
        fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap
        )
        return fallback_splitter.split_text(text)


class GeneralDocumentSplitter:
    """通用文档分块器"""

    def __init__(self, chunk_size: int = 200, chunk_overlap: int = 20):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )

    def split_text(self, text: str) -> List[str]:
        """通用文档分块"""
        return self.splitter.split_text(text)

    def split_documents(self, documents: List) -> List:
        """分割文档对象"""
        return self.splitter.split_documents(documents)
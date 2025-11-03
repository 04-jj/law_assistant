import re
from typing import List, Dict, Any
import os

from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader

from DocumentSplitter import DocumentSplitter, GeneralDocumentSplitter


class DocumentProcessor:
    """文档处理器"""

    def __init__(self):
        self.legal_splitter = DocumentSplitter(chunk_size=400, chunk_overlap=30)
        self.general_splitter = GeneralDocumentSplitter(chunk_size=200, chunk_overlap=20)

    def is_legal_document(self, file_path: str) -> bool:
        """判断是否为法律文档"""
        filename = os.path.basename(file_path)
        legal_keywords = ['法', '条例', '规定', '办法', '细则', '章程', '规范', '法律']
        filename_lower = filename.lower()

        # 根据文件名判断
        for keyword in legal_keywords:
            if keyword in filename_lower:
                return True

        # 如果文件名无法判断，可以进一步检查文件内容
        try:
            content = self._load_file_content(file_path)
            # 检查内容中是否包含法律特征
            if self._has_legal_characteristics(content):
                return True
        except:
            pass

        return False

    def _has_legal_characteristics(self, content: str) -> bool:
        """检查内容是否具有法律文档特征"""
        legal_patterns = [
            r'第[零一二三四五六七八九十百千万\d]+条',
            r'《[^》]+》',
            r'第一章|第二章|第三章|第四章|第五章|第六章|第七章|第八章|第九章|第十章',
            r'第一条|第二条|第三条|第四条|第五条'
        ]

        for pattern in legal_patterns:
            if re.search(pattern, content):
                return True
        return False

    def _load_file_content(self, file_path: str) -> str:
        """加载文件内容"""
        if file_path.lower().endswith('.txt'):
            loader = TextLoader(file_path, encoding='utf-8')
        elif file_path.lower().endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        elif file_path.lower().endswith(('.doc', '.docx')):
            loader = Docx2txtLoader(file_path)
        else:
            return ""

        documents = loader.load()
        return "\n".join([doc.page_content for doc in documents])

    def process_document(self, file_path: str) -> List[Dict[str, Any]]:
        """处理文档，自动识别类型并采用相应分块策略"""
        if self.is_legal_document(file_path):
            return self.process_legal_document(file_path)
        else:
            return self.process_general_document(file_path)

    def process_legal_document(self, file_path: str) -> List[Dict[str, Any]]:
        """处理法律文档，返回结构化的条款"""

        # 加载文档
        documents = self._load_documents(file_path)
        structured_articles = []

        for doc in documents:
            content = doc.page_content
            articles = self._extract_structured_articles(content)
            structured_articles.extend(articles)

        return structured_articles

    def process_general_document(self, file_path: str) -> List[Dict[str, Any]]:
        """处理普通文档"""
        documents = self._load_documents(file_path)
        chunks = self.general_splitter.split_documents(documents)

        structured_chunks = []
        for i, chunk in enumerate(chunks):
            structured_chunks.append({
                'document_type': 'general',
                'chunk_number': i + 1,
                'content': chunk.page_content,
                'full_text': chunk.page_content,
                'metadata': {
                    'source': 'general_document',
                    'chunk_type': 'text_segment'
                }
            })

        return structured_chunks

    def _load_documents(self, file_path: str) -> List:
        """加载文档"""
        if file_path.lower().endswith('.txt'):
            loader = TextLoader(file_path, encoding='utf-8')
        elif file_path.lower().endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        elif file_path.lower().endswith(('.doc', '.docx')):
            loader = Docx2txtLoader(file_path)
        else:
            raise ValueError(f"不支持的文件格式: {file_path}")

        return loader.load()

    def _extract_structured_articles(self, content: str) -> List[Dict[str, Any]]:
        """从文本中提取结构化的法律条款"""

        # 匹配法律名称
        law_name_match = re.search(r'《([^》]+)》', content)
        law_name = law_name_match.group(1) if law_name_match else "未知法律"

        # 匹配条款
        article_pattern = r'(第[零一二三四五六七八九十百千万\d]+条[^第]*)'
        articles = re.findall(article_pattern, content)

        structured_articles = []

        for i, article in enumerate(articles):
            # 清理条款文本
            article_clean = article.strip()

            # 提取条款编号
            article_num_match = re.search(r'第([零一二三四五六七八九十百千万\d]+)条', article_clean)
            article_num = article_num_match.group(1) if article_num_match else str(i + 1)

            # 提取条款内容（去掉编号部分）
            content_start = article_num_match.end() if article_num_match else 0
            article_content = article_clean[content_start:].strip()

            # 如果内容以"，"开头，去掉
            if article_content.startswith('，'):
                article_content = article_content[1:].strip()

            structured_articles.append({
                'law_name': law_name,
                'article_number': article_num,
                'article_content': article_content,
                'full_text': f"《{law_name}》第{article_num}条 {article_content}",
                'metadata': {
                    'source': 'legal_document',
                    'article_type': 'clause'
                }
            })

        return structured_articles
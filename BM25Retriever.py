import jieba
from rank_bm25 import BM25Okapi
from typing import List, Tuple
import os
import pickle


class BM25Retriever:
    """BM25检索器"""

    def __init__(self, bm25_index_path: str = "bm25_index.pkl"):
        self.bm25_index_path = bm25_index_path
        self.bm25 = None
        self.documents = []
        self.tokenized_docs = []

    def chinese_tokenize(self, text: str) -> List[str]:
        """中文分词"""
        return list(jieba.cut(text))

    def build_index(self, documents: List[str]):
        """构建BM25索引"""
        if not documents:
            return

        self.documents = documents
        print(f"正在构建BM25索引，文档数量: {len(documents)}")

        # 分词处理
        self.tokenized_docs = [self.chinese_tokenize(doc) for doc in documents]

        # 构建BM25模型
        self.bm25 = BM25Okapi(self.tokenized_docs)
        print("BM25索引构建完成")

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """BM25检索"""
        if self.bm25 is None or not self.documents:
            return []

        # 查询分词
        tokenized_query = self.chinese_tokenize(query)

        # BM25评分
        scores = self.bm25.get_scores(tokenized_query)

        # 获取top_k结果
        doc_scores = list(zip(self.documents, scores))
        doc_scores.sort(key=lambda x: x[1], reverse=True)

        return doc_scores[:top_k]

    def save_index(self):
        """保存BM25索引"""
        if self.bm25 is not None:
            with open(self.bm25_index_path, 'wb') as f:
                pickle.dump({
                    'bm25': self.bm25,
                    'documents': self.documents,
                    'tokenized_docs': self.tokenized_docs
                }, f)
            print(f"BM25索引已保存到: {self.bm25_index_path}")

    def load_index(self) -> bool:
        """加载BM25索引"""
        if not os.path.exists(self.bm25_index_path):
            return False

        try:
            with open(self.bm25_index_path, 'rb') as f:
                data = pickle.load(f)
                self.bm25 = data['bm25']
                self.documents = data['documents']
                self.tokenized_docs = data['tokenized_docs']
            print(f"BM25索引已从 {self.bm25_index_path} 加载")
            return True
        except Exception as e:
            print(f"加载BM25索引失败: {e}")
            return False
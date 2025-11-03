from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores.faiss import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
import os
import requests
import yaml
import numpy as np
from typing import List, Tuple
from dotenv import load_dotenv

from ConversationMemory import ConversationMemory
from DocumentProcessor import DocumentProcessor
from DocumentSplitter import GeneralDocumentSplitter

load_dotenv()


class DeepSeekApiRag:
    def __init__(self, api_key: str = None, db_path: str = None):
        # 从环境变量获取配置，如果参数为None则使用环境变量
        if api_key is None:
            api_key = os.getenv("DEEPSEEK_API_KEY")
        if db_path is None:
            db_path = os.getenv("VECTOR_DB_PATH", "law_faiss")

        # 1. 初始化嵌入模型
        print("正在加载嵌入模型...")
        embedding_model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={'device': 'cuda'},
            encode_kwargs={'normalize_embeddings': True}
        )

        # 2. 初始化DeepSeek API
        print("正在初始化DeepSeek API...")
        deepseek_base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
        deepseek_model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

        self.llm = ChatOpenAI(
            api_key=api_key,
            base_url=deepseek_base_url,
            model=deepseek_model,
        )

        # 3. 初始化向量数据库
        self.db_path = db_path
        self.vector_db = None

        # 4. 初始化文档处理器
        self.document_processor = DocumentProcessor()
        self.general_splitter = GeneralDocumentSplitter(chunk_size=200, chunk_overlap=20)

        # 5. 初始化Reranker配置
        self.reranker_api_key = os.getenv("RERANKER_API_KEY")
        self.reranker_url = os.getenv("RERANKER_BASE_URL", "https://api.siliconflow.cn/v1/rerank")
        self.reranker_model = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")

        # 6. 初始化记忆模块
        self.memory = ConversationMemory(max_history_turns=5)

        # 如果向量数据库已存在，直接加载
        if os.path.exists(db_path):
            print(f"加载已存在的向量数据库: {db_path}")
            self.load_vector_db()
        else:
            print(f"向量数据库不存在，将在添加文档时创建: {db_path}")

    def _load_prompt(self, prompt_name: str = "legal_advisor_prompt") -> str:
        """从YAML文件加载提示词模板"""
        prompts_file = "prompts.yaml"
        current_dir = os.path.dirname(os.path.abspath(__file__))
        prompts_path = os.path.join(current_dir, prompts_file)

        print(f"正在加载提示词文件: {prompts_path}")

        if not os.path.exists(prompts_path):
            raise FileNotFoundError(f"提示词文件不存在: {prompts_path}")

        try:
            with open(prompts_path, 'r', encoding='utf-8') as file:
                prompts = yaml.safe_load(file)

            if not prompts or prompt_name not in prompts:
                raise ValueError(f"提示词 '{prompt_name}' 在YAML文件中不存在")

            print(f"成功加载提示词模板: {prompt_name}")
            return prompts[prompt_name]

        except yaml.YAMLError as e:
            raise ValueError(f"YAML文件解析错误: {e}")
        except Exception as e:
            raise ValueError(f"加载提示词文件失败: {e}")

    def _get_prompt(self, prompt_name: str = "legal_advisor_prompt", **kwargs) -> str:
        """获取格式化后的提示词"""
        prompt_template = self._load_prompt(prompt_name)

        try:
            formatted_prompt = prompt_template.format(**kwargs)
            return formatted_prompt
        except KeyError as e:
            raise ValueError(f"提示词格式化错误: 缺少参数 {e}")

    def _rerank_documents(
            self,
            query: str,
            documents: List[str],
            top_k: int = 3
    ) -> List[Tuple[str, float]]:
        if not self.reranker_api_key:
            print("未设置 Reranker API 密钥，跳过重排序")
            return [(doc, 0.0) for doc in documents[:top_k]]

        headers = {
            "Authorization": f"Bearer {self.reranker_api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.reranker_model,
            "query": query,
            "documents": documents
        }

        try:
            response = requests.post(
                self.reranker_url,
                json=payload,
                headers=headers,
                timeout=30)
            response.raise_for_status()
            results = response.json().get("results", [])

            reranked = sorted(
                zip(documents, [res["relevance_score"] for res in results]),
                key=lambda x: x[1],
                reverse=True
            )
            return reranked[:top_k]
        except Exception as e:
            print(f"Reranker 调用失败: {e}")
            return [(doc, 0.0) for doc in documents[:top_k]]

    def add_documents(self, documents: List[str], save_to_disk: bool = True):
        """添加文档到向量数据库"""
        if not documents:
            return

        print(f"正在向向量数据库添加 {len(documents)} 个文档块...")

        # 手动生成嵌入向量并确保是numpy数组格式
        embeddings = self.embedding_model.embed_documents(documents)
        embeddings_array = np.array(embeddings, dtype=np.float32)

        # 检查嵌入维度是否一致
        if len(embeddings_array.shape) != 2:
            raise ValueError(f"嵌入维度不正确，期望2D数组，得到{embeddings_array.shape}")

        if self.vector_db is None:
            # 使用FAISS.from_embeddings方法
            docs = [Document(page_content=text) for text in documents]

            self.vector_db = FAISS.from_embeddings(
                text_embeddings=list(zip(documents, embeddings_array)),
                embedding=self.embedding_model,
                metadatas=[{} for _ in documents]
            )
            print(f"FAISS 数据库已初始化，包含 {len(documents)} 个文档块。")

        if save_to_disk:
            self.save_vector_db()

    def add_file_documents(self, file_path: str, save_to_disk: bool = True):
        """添加单个文件文档"""
        print(f"正在处理文档: {file_path}")

        try:
            # 使用文档处理器自动识别类型并处理
            structured_chunks = self.document_processor.process_document(file_path)

            # 准备添加到向量数据库的文本
            texts_to_add = []
            for chunk in structured_chunks:
                full_text = chunk['full_text']

                # 如果是法律文档且条款过长，进行分块
                if chunk.get('metadata', {}).get('source') == 'legal_document' and len(full_text) > 500:
                    from DocumentSplitter import DocumentSplitter
                    legal_splitter = DocumentSplitter(chunk_size=400, chunk_overlap=30)
                    sub_chunks = legal_splitter.split_text(full_text)
                    texts_to_add.extend(sub_chunks)
                else:
                    texts_to_add.append(full_text)

            print(f"从文档中提取了 {len(structured_chunks)} 个结构化块，生成 {len(texts_to_add)} 个文本块")

            # 添加到向量数据库
            self.add_documents(texts_to_add, save_to_disk)

        except Exception as e:
            print(f"文档处理失败: {e}")
            # 回退到普通分块
            self._fallback_add_documents(file_path, save_to_disk)

    def _fallback_add_documents(self, file_path: str, save_to_disk: bool = True):
        """回退到普通分块策略"""
        print(f"使用普通分块策略处理: {file_path}")

        # 加载文档
        if file_path.lower().endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        elif file_path.lower().endswith(('.doc', '.docx')):
            loader = Docx2txtLoader(file_path)
        elif file_path.lower().endswith('.txt'):
            loader = TextLoader(file_path, encoding='utf-8')
        else:
            print(f"不支持的文件格式: {file_path}")
            return

        pages = loader.load()
        documents = self.general_splitter.split_documents(pages)
        texts = [doc.page_content for doc in documents]
        self.add_documents(texts, save_to_disk)

    def add_folder_documents(self, folder_path: str, save_to_disk: bool = True):
        """添加文件夹中的所有文档（自动识别类型）"""
        supported_extensions = ('.pdf', '.doc', '.docx', '.txt')

        if not os.path.exists(folder_path):
            print(f"文件夹不存在: {folder_path}")
            return

        for filename in os.listdir(folder_path):
            if filename.lower().endswith(supported_extensions):
                file_path = os.path.join(folder_path, filename)
                print(f"正在处理文件: {file_path}")
                self.add_file_documents(file_path, save_to_disk=False)

        if save_to_disk and self.vector_db is not None:
            self.save_vector_db()

    def save_vector_db(self):
        """保存向量数据库到本地"""
        if self.vector_db is not None:
            self.vector_db.save_local(self.db_path)
            print(f"向量数据库已保存到: {self.db_path}")

    def load_vector_db(self):
        """从本地加载向量数据库"""
        self.vector_db = FAISS.load_local(
            self.db_path,
            self.embedding_model,
            allow_dangerous_deserialization=True
        )
        print(f"向量数据库已从 {self.db_path} 加载")

    def retrieve_documents(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """检索 + 重排序"""
        if self.vector_db is None:
            raise ValueError("知识库中没有文档，请先添加文档")

        docs_and_scores = self.vector_db.similarity_search_with_score(query, k=10)
        initial_docs = [doc.page_content for doc, _ in docs_and_scores]

        reranked_docs = self._rerank_documents(query, initial_docs, top_k=top_k)

        return [
            (doc, next(score for d, score in docs_and_scores if d.page_content == doc))
            for doc, _ in reranked_docs
        ]

    def generate_response_stream(self, query: str, conversation_id: str = None, top_k: int = 3,
                                 prompt_name: str = "legal_advisor_prompt"):
        """生成RAG回答（带记忆）"""
        try:
            retrieved_docs = self.retrieve_documents(query, top_k=top_k)
        except ValueError:
            retrieved_docs = []

        # 构建上下文
        context_parts = []

        # 1. 添加检索到的文档上下文
        if retrieved_docs:
            for i, (doc, score) in enumerate(retrieved_docs):
                short_doc = doc[:200] + "..." if len(doc) > 200 else doc
                context_parts.append(f"【相关文档{i + 1}】(相似度:{score:.2f}): {short_doc}")

        # 2. 添加对话记忆上下文
        conversation_history = ""
        if conversation_id:
            conversation_history = self.memory.get_formatted_history(conversation_id)
            if conversation_history and conversation_history != "无对话历史":
                context_parts.append(conversation_history)

        context = "\n\n".join(context_parts) if context_parts else "无相关上下文"

        # 构建增强的提示词
        prompt = self._get_prompt(
            prompt_name,
            query=query,
            context=context,
            conversation_history=conversation_history
        )

        # 使用流式调用
        response_stream = self.llm.stream(prompt)

        # 保存用户消息到记忆
        if conversation_id:
            self.memory.add_message(conversation_id, 'user', query)

        return {
            "stream": response_stream,
            "context": context,
            "retrieved_documents": [doc[0] for doc in retrieved_docs],
            "conversation_id": conversation_id
        }

    def save_bot_response(self, conversation_id: str, response: str):
        """保存AI回复到记忆"""
        if conversation_id:
            self.memory.add_message(conversation_id, 'assistant', response)

    def clear_conversation_memory(self, conversation_id: str):
        """清空特定对话的记忆"""
        self.memory.clear_conversation(conversation_id)

    def get_document_count(self) -> int:
        """获取向量数据库中的文档数量"""
        if self.vector_db is None:
            return 0
        return self.vector_db.index.ntotal if hasattr(self.vector_db.index, 'ntotal') else 0

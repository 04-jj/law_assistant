from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_community.vectorstores.faiss import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
import os
import requests
import yaml
import numpy as np
from typing import List, Tuple, Iterator
from dotenv import load_dotenv

# 导入transformers相关库
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
    from threading import Thread

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("警告: transformers库未安装，无法使用本地模型")

from BM25Retriever import BM25Retriever
from ConversationMemory import ConversationMemory
from DocumentProcessor import DocumentProcessor
from DocumentSplitter import GeneralDocumentSplitter

load_dotenv()


class LocalModel:
    """本地模型包装器"""

    def __init__(self, model_path: str, device: str = "cuda"):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers库未安装，请运行: pip install transformers")

        self.model_path = model_path
        self.device = device

        print(f"正在加载本地模型: {model_path}")

        # 加载tokenizer和模型
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )

        # 如果tokenizer没有pad_token，设置为eos_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 根据设备选择加载方式
        if device == "cuda" and torch.cuda.is_available():
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            print("模型运行在GPU模式")
        # else:
        #     self.model = AutoModelForCausalLM.from_pretrained(
        #         model_path,
        #         torch_dtype=torch.float32,
        #         device_map=None,
        #         trust_remote_code=True
        #     )
        #     self.model = self.model.to('cpu')
        #     print("模型运行在CPU模式")

        self.model.eval()  # 设置为评估模式
        print("本地模型加载完成")

    def stream_chat(self, prompt: str, max_new_tokens: int = 512, temperature: float = 0.7) -> Iterator[str]:
        """流式生成回复"""
        # 编码输入
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
            padding=True
        )

        # 移动输入到设备
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # 创建流式生成器
        streamer = TextIteratorStreamer(
            self.tokenizer,
            timeout=60.0,
            skip_prompt=True,
            skip_special_tokens=True
        )

        # 生成参数
        generation_kwargs = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "do_sample": temperature > 0,
            "top_p": 0.9,
            "streamer": streamer,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "repetition_penalty": 1.1
        }

        # 在单独线程中生成
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.daemon = True
        thread.start()

        # 流式返回结果
        generated_text = ""
        for new_text in streamer:
            generated_text += new_text
            yield new_text


class LocalModelWrapper:
    """本地模型包装器，模拟ChatOpenAI接口"""

    def __init__(self, model_path: str, device: str = "cuda"):
        self.model = LocalModel(model_path, device)

    def stream(self, prompt: str):
        """流式生成，模拟ChatOpenAI的stream方法"""

        class StreamResponse:
            def __init__(self, content):
                self.content = content
                self.type = "content"

        for chunk in self.model.stream_chat(prompt, max_new_tokens=512, temperature=0.7):
            yield StreamResponse(chunk)


class LocalRagSystem:
    def __init__(self, db_path: str = None):
        # 从环境变量获取配置
        if db_path is None:
            db_path = os.getenv("VECTOR_DB_PATH", "law_faiss")

        # 1. 初始化嵌入模型
        print("正在加载嵌入模型...")
        embedding_model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        # 2. 初始化本地大语言模型
        local_model_path = os.getenv("LOCAL_MODEL_PATH")
        local_device = os.getenv("LOCAL_MODEL_DEVICE", "cuda")

        if not local_model_path:
            raise ValueError("请设置 LOCAL_MODEL_PATH 环境变量指定本地模型路径")

        print("正在初始化本地模型...")
        try:
            self.llm = LocalModelWrapper(local_model_path, local_device)
            print("本地模型初始化成功")
        except Exception as e:
            raise RuntimeError(f"本地模型初始化失败: {e}")

        # 3. 初始化向量数据库
        self.db_path = db_path
        self.vector_db = None

        # 4. 初始化BM25检索器
        self.bm25_retriever = BM25Retriever("bm25_index.pkl")
        self.all_documents = []  # 存储所有文档内容

        # 5. 初始化文档处理器
        self.document_processor = DocumentProcessor()
        self.general_splitter = GeneralDocumentSplitter(chunk_size=200, chunk_overlap=20)

        # 6. 初始化Reranker配置
        self.reranker_api_key = os.getenv("RERANKER_API_KEY")
        self.reranker_url = os.getenv("RERANKER_BASE_URL", "https://api.siliconflow.cn/v1/rerank")
        self.reranker_model = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")

        # 7. 初始化记忆模块
        self.memory = ConversationMemory(max_history_turns=5)

        # 8. 检索权重配置
        self.vector_weight = float(os.getenv("VECTOR_RETRIEVAL_WEIGHT", "0.6"))
        self.bm25_weight = float(os.getenv("BM25_RETRIEVAL_WEIGHT", "0.4"))

        # 如果向量数据库已存在，直接加载
        if os.path.exists(db_path):
            print(f"加载已存在的向量数据库: {db_path}")
            self.load_vector_db()

        # 尝试加载BM25索引
        if not self.bm25_retriever.load_index():
            print("BM25索引不存在，将在添加文档时构建")
        else:
            self.all_documents = self.bm25_retriever.documents
            print(f"BM25索引加载成功，文档数量: {len(self.all_documents)}")

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
            top_k: int = 10
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
        """添加文档到向量数据库和BM25索引"""
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
        else:
            # 如果向量数据库已存在，添加新文档
            self.vector_db.add_texts(documents, embeddings=embeddings_array)

        # 添加到BM25索引
        self.all_documents.extend(documents)
        self.bm25_retriever.build_index(self.all_documents)

        if save_to_disk:
            self.save_vector_db()
            self.bm25_retriever.save_index()

        print(
            f"文档添加完成 - 向量数据库: {self.get_document_count()} 个文档, BM25索引: {self.get_bm25_document_count()} 个文档")

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

        if save_to_disk and (self.vector_db is not None or self.all_documents):
            self.save_vector_db()
            self.bm25_retriever.save_index()

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

    def hybrid_retrieve_documents(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """混合检索：向量检索 + BM25检索"""
        all_results = []

        # 1. 向量检索
        try:
            if self.vector_db is not None:
                vector_results = self.vector_db.similarity_search_with_score(query, k=top_k * 2)
                # 归一化向量检索分数
                vector_scores = [score for _, score in vector_results]
                if vector_scores:
                    max_vector_score = max(vector_scores)
                    min_vector_score = min(vector_scores)
                    for doc, score in vector_results:
                        if max_vector_score != min_vector_score:
                            normalized_score = (score - min_vector_score) / (max_vector_score - min_vector_score)
                        else:
                            normalized_score = 1.0
                        all_results.append((doc.page_content, normalized_score, "vector"))
                    print(f"向量检索返回 {len(vector_results)} 个结果")
        except Exception as e:
            print(f"向量检索失败: {e}")

        # 2. BM25检索
        try:
            bm25_results = self.bm25_retriever.search(query, top_k=top_k * 2)
            # 归一化BM25分数
            bm25_scores = [score for _, score in bm25_results]
            if bm25_scores:
                max_bm25_score = max(bm25_scores)
                min_bm25_score = min(bm25_scores)
                for doc, score in bm25_results:
                    if max_bm25_score != min_bm25_score:
                        normalized_score = (score - min_bm25_score) / (max_bm25_score - min_bm25_score)
                    else:
                        normalized_score = 1.0
                    all_results.append((doc, normalized_score, "bm25"))
                print(f"BM25检索返回 {len(bm25_results)} 个结果")
        except Exception as e:
            print(f"BM25检索失败: {e}")

        # 3. 结果融合（加权融合）
        fused_results = {}
        for doc, score, method in all_results:
            if doc not in fused_results:
                # 根据方法类型应用不同权重
                weight = self.vector_weight if method == "vector" else self.bm25_weight
                fused_results[doc] = score * weight
            else:
                # 如果同一个文档被两种方法检索到，取加权平均
                current_weight = self.vector_weight if method == "vector" else self.bm25_weight
                fused_results[doc] = (fused_results[doc] + score * current_weight) / 2

        # 4. 排序并返回top_k
        sorted_results = sorted(fused_results.items(), key=lambda x: x[1], reverse=True)

        final_results = [(doc, score) for doc, score in sorted_results[:top_k * 2]]
        print(f"混合检索融合后返回 {len(final_results)} 个结果")

        return final_results

    def retrieve_documents(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """混合检索 + 重排序"""
        # 使用混合检索获取更多候选文档
        hybrid_results = self.hybrid_retrieve_documents(query, top_k=top_k * 2)

        if not hybrid_results:
            print("混合检索未返回任何结果")
            return []

        initial_docs = [doc for doc, _ in hybrid_results]

        # 使用reranker进行精细排序
        reranked_docs = self._rerank_documents(query, initial_docs, top_k=top_k)

        print(f"重排序后返回 {len(reranked_docs)} 个最终结果")
        return reranked_docs

    def generate_response_stream(self, query: str, conversation_id: str = None, top_k: int = 20,
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

        # 使用本地模型流式调用
        try:
            response_stream = self.llm.stream(prompt)
        except Exception as e:
            print(f"本地模型调用失败: {e}")

            # 返回错误响应
            def error_stream():
                yield "抱歉，本地模型调用出现错误，请检查模型路径和设备配置。"

            response_stream = error_stream()

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

    def get_bm25_document_count(self) -> int:
        """获取BM25索引中的文档数量"""
        return len(self.all_documents)

    def get_retrieval_stats(self) -> dict:
        """获取检索统计信息"""
        return {
            "vector_documents": self.get_document_count(),
            "bm25_documents": self.get_bm25_document_count(),
            "vector_weight": self.vector_weight,
            "bm25_weight": self.bm25_weight,
            "reranker_enabled": bool(self.reranker_api_key)
        }
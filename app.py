from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, Response
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_sqlalchemy import SQLAlchemy
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from model_utils import DeepSeekApiRag
import os
from datetime import datetime
import uuid
from dotenv import load_dotenv
from werkzeug.security import generate_password_hash, check_password_hash

# 加载环境变量
load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///user.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = os.path.join(app.root_path, 'uploads')
app.config['KNOWLEDGE_BASE_FOLDER'] = os.path.join(app.root_path, 'knowledge_base')
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

# 确保上传文件夹和知识库文件夹存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['KNOWLEDGE_BASE_FOLDER'], exist_ok=True)

# 初始化数据库和登录管理
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message = ''


# 用户模型
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    phone = db.Column(db.String(20), unique=True, nullable=False)
    username = db.Column(db.String(50), nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    role = db.Column(db.String(20), nullable=False, default='user')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # 关联对话、知识库和上传文档
    chats = db.relationship('Chat', backref='user', lazy=True, cascade="all, delete-orphan")
    knowledge_bases = db.relationship('KnowledgeBase', backref='user', lazy=True, cascade="all, delete-orphan")
    uploaded_documents = db.relationship('UploadedDocument', backref='user', lazy=True, cascade="all, delete-orphan")

    def set_password(self, password):
        """设置密码哈希"""
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        """验证密码"""
        return check_password_hash(self.password_hash, password)


# 知识库模型
class KnowledgeBase(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # 关联上传文档
    documents = db.relationship('UploadedDocument', backref='knowledge_base', lazy=True, cascade="all, delete-orphan")


# Chat模型
class Chat(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    title = db.Column(db.String(100), nullable=False, default='新对话')
    created_at = db.Column(db.DateTime, default=datetime.now)
    updated_at = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)
    knowledge_base_id = db.Column(db.Integer, db.ForeignKey('knowledge_base.id'), nullable=True)

    # 关联消息和知识库
    messages = db.relationship('Message', backref='chat', lazy=True, cascade="all, delete-orphan")
    knowledge_base = db.relationship('KnowledgeBase', backref='chats')


# 消息模型
class Message(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    chat_id = db.Column(db.Integer, db.ForeignKey('chat.id'), nullable=False)
    role = db.Column(db.String(10), nullable=False)  # 'user' 或 'bot'
    content = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.now)


# 上传文档模型
class UploadedDocument(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    knowledge_base_id = db.Column(db.Integer, db.ForeignKey('knowledge_base.id'), nullable=True)
    filename = db.Column(db.String(255), nullable=False)  # 原始文件名
    file_path = db.Column(db.String(512), nullable=False)  # 存储路径
    file_type = db.Column(db.String(50), nullable=False)  # 文件类型
    file_size = db.Column(db.Integer, nullable=False)  # 文件大小
    uploaded_at = db.Column(db.DateTime, default=datetime.now)  # 上传时间


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# 初始化 RAG 模型
# api_key = os.getenv("LOCAL_MODEL_PATH")
api_key = os.getenv("DEEPSEEK_API_KEY")
db_path = os.getenv("VECTOR_DB_PATH", "law_faiss")


def initialize_vector_database():
    """初始化向量数据库，自动处理knowledge_base文件夹"""
    global rag_model

    # 初始化RAG模型
    rag_model = DeepSeekApiRag(api_key, db_path)

    # 检查向量数据库是否已存在
    if not os.path.exists(db_path):
        print("向量数据库不存在，开始构建...")

        all_texts = []

        # 首先处理knowledge_base文件夹
        knowledge_base_folder = app.config['KNOWLEDGE_BASE_FOLDER']
        if os.path.exists(knowledge_base_folder) and os.listdir(knowledge_base_folder):
            print(f"正在处理knowledge_base文件夹: {knowledge_base_folder}")
            rag_model.add_folder_documents(knowledge_base_folder)
            print("knowledge_base文件夹中的文档已添加到向量数据库")
        else:
            print(f"knowledge_base文件夹为空或不存在: {knowledge_base_folder}")

        # 然后处理已上传的文档（从数据库记录）
        with app.app_context():
            all_docs = UploadedDocument.query.all()
            if all_docs:
                print("正在处理已上传的文档...")
                for doc in all_docs:
                    if os.path.exists(doc.file_path):
                        try:
                            if doc.file_path.lower().endswith('.pdf'):
                                loader = PyPDFLoader(doc.file_path)
                            elif doc.file_path.lower().endswith(('.doc', '.docx')):
                                loader = Docx2txtLoader(doc.file_path)
                            elif doc.file_path.lower().endswith('.txt'):
                                loader = TextLoader(doc.file_path, encoding='utf-8')
                            else:
                                continue

                            pages = loader.load()
                            documents = rag_model.text_splitter.split_documents(pages)
                            texts = [doc.page_content for doc in documents]
                            all_texts.extend(texts)
                            print(f"已读取文档: {doc.filename}, {len(texts)} 个文本块")
                        except Exception as e:
                            print(f"处理文档 {doc.filename} 失败: {e}")

                # 一次性添加所有文本
                if all_texts:
                    rag_model.add_documents(all_texts)
                    print(f"已上传文档处理完成，共 {len(all_texts)} 个文本块")

        print(f"向量数据库构建完成，总文档数: {rag_model.get_document_count()}")
        print(f"BM25索引构建完成，总文档数: {rag_model.get_bm25_document_count()}")
    else:
        print("向量数据库已存在，跳过初始化构建")


# 创建数据库表和初始化向量数据库
with app.app_context():
    db.create_all()
    initialize_vector_database()


# 路由定义
@app.route('/')
@login_required
def home():
    return render_template('index.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('home'))

    if request.method == 'POST':
        phone = request.form.get('phone')
        username = request.form.get('username')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        role = request.form.get('role', 'user')

        # 验证输入
        if not all([phone, username, password, confirm_password]):  # 添加用户名验证
            flash('请填写所有必填字段', 'error')
            return redirect(url_for('register'))

        if len(password) < 6:
            flash('密码长度至少为6位', 'error')
            return redirect(url_for('register'))

        if password != confirm_password:
            flash('两次输入的密码不一致', 'error')
            return redirect(url_for('register'))

        # 检查手机号是否已存在
        if User.query.filter_by(phone=phone).first():
            flash('该手机号已注册', 'error')
            return redirect(url_for('register'))

        # 检查用户名是否已存在
        if User.query.filter_by(username=username).first():
            flash('该用户名已存在', 'error')
            return redirect(url_for('register'))

        # 创建新用户
        new_user = User(phone=phone, username=username, role=role)
        new_user.set_password(password)  # 设置密码

        db.session.add(new_user)
        db.session.commit()

        flash('注册成功，请登录', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))

    if request.method == 'POST':
        phone = request.form.get('identifier')
        password = request.form.get('password')
        remember = True if request.form.get('remember') else False

        # 查找用户
        user = User.query.filter_by(phone=phone).first()

        if not user:
            flash('用户不存在', 'error')
            return redirect(url_for('login'))

        if not user.check_password(password):  # 现在这个方法已定义
            flash('手机号或密码错误', 'error')
            return redirect(url_for('login'))

        login_user(user, remember=remember)
        return redirect(url_for('home'))

    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


# 知识库管理路由
@app.route('/knowledge-bases')
@login_required
def knowledge_bases():
    # 获取当前用户的所有知识库
    kb_list = KnowledgeBase.query.filter_by(user_id=current_user.id).order_by(KnowledgeBase.updated_at.desc()).all()
    return render_template('knowledge_base.html', knowledge_bases=kb_list)


@app.route('/knowledge-base/create', methods=['GET', 'POST'])
@login_required
def create_knowledge_base():
    if request.method == 'POST':
        name = request.form.get('name')
        description = request.form.get('description')

        if not name:
            flash('请输入知识库名称', 'error')
            return redirect(url_for('create_knowledge_base'))

        # 创建新知识库
        new_kb = KnowledgeBase(
            user_id=current_user.id,
            name=name,
            description=description
        )
        db.session.add(new_kb)
        db.session.commit()

        flash('知识库创建成功', 'success')
        return redirect(url_for('knowledge_bases'))

    return render_template('create_knowledge_base.html')


@app.route('/knowledge-base/<int:kb_id>/edit', methods=['GET', 'POST'])
@login_required
def edit_knowledge_base(kb_id):
    kb = KnowledgeBase.query.filter_by(id=kb_id, user_id=current_user.id).first()
    if not kb:
        flash('知识库不存在或无权访问', 'error')
        return redirect(url_for('knowledge_bases'))

    if request.method == 'POST':
        name = request.form.get('name')
        description = request.form.get('description')

        if not name:
            flash('请输入知识库名称', 'error')
            return redirect(url_for('edit_knowledge_base', kb_id=kb_id))

        kb.name = name
        kb.description = description
        db.session.commit()

        flash('知识库更新成功', 'success')
        return redirect(url_for('knowledge_bases'))

    return render_template('edit_knowledge_base.html', knowledge_base=kb)


@app.route('/knowledge-base/<int:kb_id>/delete', methods=['POST'])
@login_required
def delete_knowledge_base(kb_id):
    kb = KnowledgeBase.query.filter_by(id=kb_id, user_id=current_user.id).first()
    if not kb:
        flash('知识库不存在或无权访问', 'error')
        return redirect(url_for('knowledge_bases'))

    # 删除关联的文档和文件
    for doc in kb.documents:
        if os.path.exists(doc.file_path):
            try:
                os.remove(doc.file_path)
            except Exception as e:
                print(f"删除文件失败: {e}")

    # 从数据库中删除知识库及其文档
    db.session.delete(kb)
    db.session.commit()

    flash('知识库已删除', 'success')
    return redirect(url_for('knowledge_bases'))


# 文档上传路由
@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload_document():
    # 只有专家和管理员可以上传文件
    if current_user.role not in ['expert', 'admin']:
        flash('您没有权限上传文档', 'error')
        return redirect(url_for('home'))

    if request.method == 'POST':
        # 检查是否有文件上传
        if 'file' not in request.files:
            flash('请选择要上传的文件', 'error')
            return redirect(request.url)

        file = request.files['file']
        kb_id = request.form.get('knowledge_base_id')

        # 检查文件名是否为空
        if file.filename == '':
            flash('请选择要上传的文件', 'error')
            return redirect(request.url)

        # 检查文件格式
        allowed_extensions = {'pdf', 'docx', 'txt'}
        file_ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
        if file_ext not in allowed_extensions:
            flash('只支持上传PDF、DOCX和TXT格式的文件', 'error')
            return redirect(request.url)

        # 检查知识库是否存在
        if kb_id:
            kb = KnowledgeBase.query.filter_by(id=kb_id, user_id=current_user.id).first()
            if not kb:
                flash('知识库不存在或无权访问', 'error')
                return redirect(request.url)

        # 生成唯一文件名并保存
        filename = f"{uuid.uuid4()}.{file_ext}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # 记录上传信息到数据库
        new_doc = UploadedDocument(
            user_id=current_user.id,
            knowledge_base_id=kb_id if kb_id else None,
            filename=file.filename,
            file_path=file_path,
            file_type=file_ext,
            file_size=os.path.getsize(file_path)
        )
        db.session.add(new_doc)
        db.session.commit()

        try:
            # 将文档添加到向量数据库
            rag_model.add_file_documents(file_path)
            flash('文件上传成功并已添加到向量数据库', 'success')
        except Exception as e:
            # 如果添加到向量数据库失败，删除已保存的文件和数据库记录
            if os.path.exists(file_path):
                os.remove(file_path)
            db.session.delete(new_doc)
            db.session.commit()
            flash(f'文件上传失败: {str(e)}', 'error')
            return redirect(request.url)

        return redirect(url_for('upload_document'))

    # 获取当前用户的知识库和上传记录
    knowledge_bases = KnowledgeBase.query.filter_by(user_id=current_user.id).order_by(KnowledgeBase.name).all()
    uploaded_docs = UploadedDocument.query.filter_by(user_id=current_user.id).order_by(
        UploadedDocument.uploaded_at.desc()).all()
    return render_template('upload.html', knowledge_bases=knowledge_bases, uploaded_docs=uploaded_docs)


# 删除上传文档路由
@app.route('/upload/<int:doc_id>/delete', methods=['POST'])
@login_required
def delete_uploaded_document(doc_id):
    doc = UploadedDocument.query.filter_by(id=doc_id, user_id=current_user.id).first()
    if not doc:
        flash('文档不存在', 'error')
        return redirect(url_for('upload_document'))

    try:
        # 删除文件
        if os.path.exists(doc.file_path):
            os.remove(doc.file_path)

        # 从数据库中删除记录
        db.session.delete(doc)
        db.session.commit()

        # 重新构建向量数据库（因为FAISS不支持删除单个文档）
        flash('文档已删除，下次启动时将重新构建向量数据库', 'success')

    except Exception as e:
        db.session.rollback()
        flash(f'删除文档失败: {str(e)}', 'error')

    return redirect(url_for('upload_document'))


# 流式对话路由
@app.route('/ask_stream', methods=['POST', 'GET'])
@login_required
def ask_stream():
    """专门的流式对话接口"""
    # 处理参数
    if request.method == 'GET':
        user_input = request.args.get('user_input')
        chat_id = request.args.get('chat_id')
        kb_id = request.args.get('knowledge_base_id')
    else:
        user_input = request.form.get('user_input')
        chat_id = request.form.get('chat_id')
        kb_id = request.form.get('knowledge_base_id')

    if not user_input or not chat_id:
        return jsonify({'error': '缺少必要参数'}), 400

    print(f"用户提问: {user_input}")

    # 使用chat_id作为对话记忆的标识
    conversation_id = f"chat_{chat_id}"

    # 根据用户角色和选择的知识库使用不同的知识库
    if current_user.role in ['expert', 'admin'] and kb_id:
        kb = KnowledgeBase.query.filter_by(id=kb_id, user_id=current_user.id).first()
        if kb:
            doc_paths = [doc.file_path for doc in kb.documents]
            temp_rag = DeepSeekApiRag(api_key, db_path)
            for path in doc_paths:
                if os.path.exists(path):
                    temp_rag.add_file_documents(path)
            result = temp_rag.generate_response_stream(
                user_input,
                conversation_id=conversation_id
            )
        else:
            result = rag_model.generate_response_stream(
                user_input,
                conversation_id=conversation_id
            )
    else:
        result = rag_model.generate_response_stream(
            user_input,
            conversation_id=conversation_id
        )

    # 先保存用户消息到数据库
    try:
        user_message = Message(
            chat_id=chat_id,
            role='user',
            content=user_input
        )
        db.session.add(user_message)
        db.session.commit()
        print(f"用户消息已保存到数据库，chat_id: {chat_id}")
    except Exception as e:
        print(f"保存用户消息失败: {e}")
        db.session.rollback()

    def generate():
        full_response = ""
        try:
            # 流式输出响应
            for chunk in result['stream']:
                content = chunk.content
                full_response += content
                cleaned_content = content.replace('\n', '\\n').replace('"', '\\"')
                yield f"data: {{\"content\": \"{cleaned_content}\"}}\n\n"

            # 保存AI回复到记忆
            rag_model.save_bot_response(conversation_id, full_response)

            # 在应用上下文中保存机器人回复到数据库
            with app.app_context():
                try:
                    bot_message = Message(
                        chat_id=chat_id,
                        role='bot',
                        content=full_response
                    )
                    db.session.add(bot_message)

                    # 更新对话时间和知识库
                    chat = Chat.query.get(chat_id)
                    if chat:
                        chat.updated_at = datetime.utcnow()
                        if kb_id:
                            chat.knowledge_base_id = kb_id

                    db.session.commit()
                    print(f"机器人回复已保存到数据库，长度: {len(full_response)}")
                except Exception as e:
                    print(f"保存机器人回复失败: {e}")
                    db.session.rollback()

            yield "data: {\"done\": true}\n\n"

        except Exception as e:
            print(f"流式响应错误: {e}")
            # 保存错误消息到记忆
            error_msg = f"抱歉，生成回复时出现错误: {str(e)}"
            rag_model.save_bot_response(conversation_id, error_msg)

            # 在应用上下文中保存错误消息到数据库
            with app.app_context():
                try:
                    error_message = Message(
                        chat_id=chat_id,
                        role='bot',
                        content=error_msg
                    )
                    db.session.add(error_message)

                    chat = Chat.query.get(chat_id)
                    if chat:
                        chat.updated_at = datetime.utcnow()

                    db.session.commit()
                    print("错误消息已保存到数据库")
                except Exception as db_error:
                    print(f"保存错误消息失败: {db_error}")
                    db.session.rollback()

            yield f"data: {{\"error\": \"{error_msg}\"}}\n\n"
            yield "data: {\"done\": true}\n\n"

    return Response(generate(), mimetype='text/event-stream')


# 清空对话记忆路由
@app.route('/api/chats/<int:chat_id>/clear_memory', methods=['POST'])
@login_required
def clear_chat_memory(chat_id):
    """清空特定对话的记忆"""
    chat = Chat.query.filter_by(id=chat_id, user_id=current_user.id).first()
    if not chat:
        return jsonify({'error': '对话不存在'}), 404

    conversation_id = f"chat_{chat_id}"
    rag_model.clear_conversation_memory(conversation_id)

    return jsonify({'success': True, 'message': '对话记忆已清空'})


# 对话相关API
@app.route('/api/chats')
@login_required
def get_chats():
    """获取当前用户的所有对话"""
    chats = Chat.query.filter_by(user_id=current_user.id).order_by(Chat.updated_at.desc()).all()
    return jsonify([{
        'id': chat.id,
        'title': chat.title,
        'created_at': chat.created_at.isoformat(),
        'updated_at': chat.updated_at.isoformat(),
        'knowledge_base_id': chat.knowledge_base_id,
        'knowledge_base_name': chat.knowledge_base.name if chat.knowledge_base else None
    } for chat in chats])


@app.route('/api/chats/<int:chat_id>')
@login_required
def get_chat_messages(chat_id):
    """获取特定对话的消息"""
    chat = Chat.query.filter_by(id=chat_id, user_id=current_user.id).first()
    if not chat:
        return jsonify({'error': '对话不存在'}), 404

    messages = Message.query.filter_by(chat_id=chat_id).order_by(Message.created_at).all()
    return jsonify({
        'id': chat.id,
        'title': chat.title,
        'knowledge_base_id': chat.knowledge_base_id,
        'knowledge_base_name': chat.knowledge_base.name if chat.knowledge_base else None,
        'messages': [{
            'id': msg.id,
            'role': msg.role,
            'content': msg.content,
            'created_at': msg.created_at.isoformat()
        } for msg in messages]
    })


@app.route('/api/chats', methods=['POST'])
@login_required
def create_chat():
    """创建新对话"""
    data = request.json
    title = data.get('title', '新对话')
    kb_id = data.get('knowledge_base_id')

    new_chat = Chat(user_id=current_user.id, title=title, knowledge_base_id=kb_id)
    db.session.add(new_chat)
    db.session.commit()

    # 添加初始消息
    initial_msg = Message(
        chat_id=new_chat.id,
        role='bot',
        content='您好！我是DeepSeek智能助手，请问有什么可以帮您的吗？'
    )
    db.session.add(initial_msg)
    db.session.commit()

    return jsonify({
        'id': new_chat.id,
        'title': new_chat.title,
        'knowledge_base_id': new_chat.knowledge_base_id,
        'created_at': new_chat.created_at.isoformat()
    }), 201


@app.route('/api/chats/<int:chat_id>', methods=['PUT'])
@login_required
def update_chat(chat_id):
    """更新对话标题或知识库"""
    chat = Chat.query.filter_by(id=chat_id, user_id=current_user.id).first()
    if not chat:
        return jsonify({'error': '对话不存在'}), 404

    data = request.json
    if 'title' in data:
        chat.title = data['title']
    if 'knowledge_base_id' in data:
        chat.knowledge_base_id = data['knowledge_base_id']

    chat.updated_at = datetime.utcnow()
    db.session.commit()

    return jsonify({
        'id': chat.id,
        'title': chat.title,
        'knowledge_base_id': chat.knowledge_base_id,
        'updated_at': chat.updated_at.isoformat()
    })


@app.route('/api/chats/<int:chat_id>', methods=['DELETE'])
@login_required
def delete_chat(chat_id):
    """删除对话"""
    chat = Chat.query.filter_by(id=chat_id, user_id=current_user.id).first()
    if not chat:
        return jsonify({'error': '对话不存在'}), 404

    # 同时清空对话记忆
    conversation_id = f"chat_{chat_id}"
    rag_model.clear_conversation_memory(conversation_id)

    db.session.delete(chat)
    db.session.commit()
    return jsonify({'success': True})


# 知识库相关API
@app.route('/api/knowledge-bases')
@login_required
def get_knowledge_bases():
    """获取当前用户的所有知识库"""
    kb_list = KnowledgeBase.query.filter_by(user_id=current_user.id).order_by(KnowledgeBase.updated_at.desc()).all()
    return jsonify([{
        'id': kb.id,
        'name': kb.name,
        'description': kb.description,
        'created_at': kb.created_at.isoformat(),
        'updated_at': kb.updated_at.isoformat(),
        'document_count': len(kb.documents)
    } for kb in kb_list])


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
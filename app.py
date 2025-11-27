import os
import sqlite3
import uuid
from datetime import datetime
from flask import Flask, request, jsonify, session, redirect, url_for, render_template, g, send_from_directory
from functools import wraps
import re
import sys 
from pathlib import Path
# Import your custom tools
try:
    from data_preparer import DocumentExtractor
    from qdrant_handle import EnhancedHybridQdrantHandler
    from llm import SLM_TALWAR
except ImportError:
    print("="*50)
    print("ERROR: Could not import custom modules (DocumentExtractor, EnhancedHybridQdrantHandler, SLM_TALWAR).")
    print("Please ensure 'Tools' directory is in the same folder as app.py and all dependencies are installed.")
    print("="*50)
    sys.exit(1)

# --- Configuration ---
DATABASE = 'talwar_chat_app.db'
UPLOAD_FOLDER = 'uploaded_docs'
STATIC_FOLDER = 'static'
ALLOWED_EXTENSIONS = {'pdf', 'txt', 'md', 'text'}
COLLECTION_NAME = "RAG"

# Path to your BGE-M3 model directory (Ensure this exists)
# base_dir = Path(__file__).resolve().parent
MODEL_PATH = os.path.join( "model", "bge-m3")


app = Flask(__name__, static_folder=STATIC_FOLDER)
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', 'a_very_strong_development_secret_key_123!@#')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- Global Component Initialization ---
print("Initializing AI components...")
try:
    doc_extractor = DocumentExtractor()
    
    # Updated: Initialize with specific model path for BGE-M3
    handler = EnhancedHybridQdrantHandler(model_path=MODEL_PATH)
    
    slm = SLM_TALWAR()

    # Setup Qdrant collection on startup
    print(f"Setting up Qdrant collection: {COLLECTION_NAME}")
    handler.setup_hybrid_collection(COLLECTION_NAME)
    print("AI components initialized successfully.")
except Exception as e:
    print(f"FATAL ERROR during AI component initialization: {e}")
    print("Please check model paths, Qdrant connection, and dependencies.")
    exit(1)

# Ensure upload and static directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)
os.makedirs(os.path.join(STATIC_FOLDER, 'images'), exist_ok=True)

# --- Database Setup (SQLite) ---

def get_db():
    """Get a database connection from the global context 'g'."""
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
        db.row_factory = sqlite3.Row
    return db

@app.teardown_appcontext
def close_connection(exception):
    """Close the database connection at the end of the request."""
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

def init_db():
    """Initialize the database schema."""
    with app.app_context():
        db = get_db()
        cursor = db.cursor()
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS chats (
            chat_id TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL,
            title TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            is_rag_chat BOOLEAN DEFAULT 0,
            rag_context_file TEXT,
            is_global_rag BOOLEAN DEFAULT 0,
            FOREIGN KEY (user_id) REFERENCES users (user_id)
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            message_id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id TEXT NOT NULL,
            role TEXT NOT NULL,  -- 'user' or 'assistant'
            content TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (chat_id) REFERENCES chats (chat_id) ON DELETE CASCADE
        )
        ''')
        
        # Add default users
        for user, pwd in [('admin', 'admin'), ('user', 'user')]:
            cursor.execute("SELECT * FROM users WHERE username = ?", (user,))
            if not cursor.fetchone():
                print(f"Creating default '{user}' user.")
                cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (user, pwd))
            
        db.commit()
        print("Database initialized.")

# --- Authentication ---

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        
        db = get_db()
        user = db.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        
        if user and user['password'] == password:
            session.clear()
            session['user_id'] = user['user_id']
            session['username'] = user['username']
            return jsonify({'success': True})
        else:
            return jsonify({'success': False, 'message': 'Invalid username or password'}), 401
            
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    session.clear()
    return redirect(url_for('login'))

# --- Main Application Routes ---

@app.route('/')
@login_required
def index():
    return render_template('index.html', username=session['username'])

@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory(app.static_folder, path)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Chat API Endpoints ---

@app.route('/get_history', methods=['GET'])
@login_required
def get_history():
    user_id = session['user_id']
    current_chat_id = session.get('current_chat_id')
    
    db = get_db()
    chats = db.execute(
        "SELECT * FROM chats WHERE user_id = ? ORDER BY created_at DESC", (user_id,)
    ).fetchall()
    
    history_list = []
    for chat in chats:
        history_list.append({
            'chat_id': chat['chat_id'],
            'title': chat['title'],
            'is_current': chat['chat_id'] == current_chat_id,
            'is_rag_chat': chat['is_rag_chat'],
            'is_global_rag': chat['is_global_rag']
        })
        
    return jsonify({'history': history_list})

@app.route('/new_chat', methods=['POST'])
@login_required
def new_chat():
    user_id = session['user_id']
    new_chat_id = str(uuid.uuid4())
    
    db = get_db()
    db.execute(
        "INSERT INTO chats (chat_id, user_id, title, is_rag_chat, is_global_rag) VALUES (?, ?, ?, 0, 0)",
        (new_chat_id, user_id, "New Chat")
    )
    db.commit()
    
    session['current_chat_id'] = new_chat_id
    return jsonify({'success': True, 'chat_id': new_chat_id})

@app.route('/new_rag_chat', methods=['POST'])
@login_required
def new_rag_chat():
    user_id = session['user_id']
    new_chat_id = str(uuid.uuid4())
    
    db = get_db()
    db.execute(
        "INSERT INTO chats (chat_id, user_id, title, is_rag_chat, is_global_rag) VALUES (?, ?, ?, 0, 1)",
        (new_chat_id, user_id, "Global RAG Chat")
    )
    db.commit()
    
    session['current_chat_id'] = new_chat_id
    return jsonify({'success': True, 'chat_id': new_chat_id})

@app.route('/load_chat/<chat_id>', methods=['GET'])
@login_required
def load_chat(chat_id):
    user_id = session['user_id']
    db = get_db()
    
    chat_info = db.execute(
        "SELECT * FROM chats WHERE chat_id = ? AND user_id = ?", (chat_id, user_id)
    ).fetchone()
    
    if not chat_info:
        return jsonify({'success': False, 'error': 'Chat not found'}), 404
        
    session['current_chat_id'] = chat_id
    
    messages = db.execute(
        "SELECT role, content FROM messages WHERE chat_id = ? ORDER BY timestamp ASC", (chat_id,)
    ).fetchall()
    
    message_list = [{'role': msg['role'], 'content': msg['content']} for msg in messages]
    
    return jsonify({
        'success': True,
        'messages': message_list,
        'title': chat_info['title'],
        'is_global_rag': chat_info['is_global_rag'],
        'rag_context_file': chat_info['rag_context_file']
    })

@app.route('/delete_chat/<chat_id>', methods=['DELETE'])
@login_required
def delete_chat(chat_id):
    user_id = session['user_id']
    db = get_db()
    
    chat = db.execute("SELECT * FROM chats WHERE chat_id = ? AND user_id = ?", (chat_id, user_id)).fetchone()
    
    if chat:
        db.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))
        db.execute("DELETE FROM chats WHERE chat_id = ?", (chat_id,))
        db.commit()
        
        if session.get('current_chat_id') == chat_id:
            session.pop('current_chat_id', None)
            
        return jsonify({'success': True})
    else:
        return jsonify({'success': False, 'error': 'Chat not found'}), 404

@app.route('/upload_doc', methods=['POST'])
@login_required
def upload_doc():
    if session.get('username') != 'admin':
        return jsonify({'success': False, 'message': 'Permission denied.'}), 403
    
    if 'file' not in request.files:
        return jsonify({'success': False, 'message': 'No file part'}), 400
        
    file = request.files['file']
    
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'success': False, 'message': 'Invalid file'}), 400
        
    # Sanitize filename
    filename = re.sub(r'[^a-zA-Z0-9_.-]', '_', file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    try:
        file.save(file_path)
        
        print(f"Processing file: {file_path}")
        extracted_text = doc_extractor.extract(file_path)
        chunks = doc_extractor.chunk_with_metadata(extracted_text, file_path)
        
        # Updated: calling standard insert_chunks (auto-generates sparse/dense with BGE-M3)
        print(f"Inserting {len(chunks)} chunks into Qdrant...")
        handler.insert_chunks(COLLECTION_NAME, chunks)
        print("File processing complete.")

        user_id = session['user_id']
        new_chat_id = str(uuid.uuid4())
        title = f"Chat with {filename}"
        
        db = get_db()
        db.execute(
            "INSERT INTO chats (chat_id, user_id, title, is_rag_chat, rag_context_file, is_global_rag) VALUES (?, ?, ?, 1, ?, 0)",
            (new_chat_id, user_id, title, filename)
        )
        db.commit()
        
        session['current_chat_id'] = new_chat_id
        
        return jsonify({'success': True, 'chat_id': new_chat_id, 'title': title})

    except Exception as e:
        print(f"Error during file processing: {e}")
        return jsonify({'success': False, 'message': f'Error processing file: {e}'}), 500

@app.route('/chat', methods=['POST'])
@login_required
def chat():
    user_id = session['user_id']
    data = request.get_json()
    user_message = data.get('message')
    
    if not user_message:
        return jsonify({'error': 'No message provided'}), 400
        
    current_chat_id = session.get('current_chat_id')
    
    if not current_chat_id:
        new_chat()
        current_chat_id = session.get('current_chat_id')

    db = get_db()
    
    db.execute(
        "INSERT INTO messages (chat_id, role, content) VALUES (?, 'user', ?)",
        (current_chat_id, user_message)
    )
    db.commit()
    
    chat_info = db.execute("SELECT * FROM chats WHERE chat_id = ?", (current_chat_id,)).fetchone()
    messages_from_db = db.execute(
        "SELECT role, content FROM messages WHERE chat_id = ? ORDER BY timestamp ASC",
        (current_chat_id,)
    ).fetchall()
    
    conversation_history = [{'role': msg['role'], 'content': msg['content']} for msg in messages_from_db]
    
    response_text = ""
    try:
        if chat_info['is_rag_chat']:
            # --- SINGLE FILE RAG ---
            filename = chat_info['rag_context_file']
            print(f"Performing single-file RAG search (Target: {filename})...")
            
            # 1. Fetch more candidates
            all_results = handler.search(COLLECTION_NAME, user_message, limit=20)
            
            # 2. Filter list by filename payload (Requires OBJECT notation here)
            filtered_results = [
                res for res in all_results 
                if res.payload.get('filename') == filename
            ]
            
            # 3. Take top 5
            final_results_objs = filtered_results[:5]
            
            # 4. CONVERT TO DICTS for LLM (LLM expects subscriptable 'dict')
            final_results_dicts = [
                {"payload": res.payload, "score": res.score, "id": res.id} 
                for res in final_results_objs
            ]
            
            print(f"Found {len(final_results_dicts)} relevant chunks.")
            response_text = slm.generate_rag_with_history(user_message, conversation_history, final_results_dicts)
            
            # --- SOURCE APPEND LOGIC (SINGLE FILE) ---
            if final_results_dicts:
                response_text += f"\n\n**Source:** {filename}"
            
        elif chat_info['is_global_rag']:
            # --- GLOBAL RAG ---
            print("Performing global RAG search...")
            
            # 1. Search
            results_objs = handler.search(COLLECTION_NAME, user_message, limit=5)
            
            # 2. CONVERT TO DICTS for LLM
            results_dicts = [
                {"payload": res.payload, "score": res.score, "id": res.id} 
                for res in results_objs
            ]

            print(f"Found {len(results_dicts)} relevant chunks.")
            response_text = slm.generate_rag_with_history(user_message, conversation_history, results_dicts)
            
            # --- SOURCE APPEND LOGIC (GLOBAL) ---
            # Extract unique filenames from the results used for generation
            unique_sources = list(set([res['payload'].get('filename', 'Unknown') for res in results_dicts]))
            if unique_sources:
                response_text += f"\n\n**Sources:** {', '.join(unique_sources)}"
            
        else:
            # --- STANDARD CHAT ---
            print("Performing standard chat generation...")
            response_text = slm.generate_with_history(user_message, conversation_history)
            
    except Exception as e:
        print(f"Error during LLM generation: {e}")
        import traceback
        traceback.print_exc()
        response_text = f"I'm sorry, but I encountered an error: {str(e)}"

    db.execute(
        "INSERT INTO messages (chat_id, role, content) VALUES (?, 'assistant', ?)",
        (current_chat_id, response_text)
    )
    
    # Title Generation
    if len(conversation_history) == 1 and chat_info['title'] == 'New Chat':
        try:
            summary_history = [
                {'role': 'user', 'content': user_message},
                {'role': 'assistant', 'content': response_text}
            ]
            title = slm.summarize_conversation(summary_history)
            title = re.sub(r'["\']', '', title).strip()[:50]
            if not title: title = user_message[:50]
            
            db.execute("UPDATE chats SET title = ? WHERE chat_id = ?", (title, current_chat_id))
        except Exception as e:
            print(f"Title generation failed: {e}")

    db.commit()
    return jsonify({'response': response_text, 'chat_id': current_chat_id})

if __name__ == '__main__':
    init_db()
    print("Starting Flask application...")
    print("Flask app running on http://0.0.0.0:5000")
    app.run(debug=True, port=5000, host='0.0.0.0')
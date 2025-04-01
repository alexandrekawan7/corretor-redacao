from dotenv import load_dotenv

load_dotenv()

from flask import Flask, render_template, request, jsonify
import fitz  # PyMuPDF
import cv2
import pytesseract
import os
from openai import OpenAI
from werkzeug.utils import secure_filename

GPT_CONTEXT = """
    Eu fornecerei uma redação escrita por um usuário qualquer, infira o tema e corrija a redação, me retornando
    apenas as notas em cada competência, sem texto extra: 
"""

client = OpenAI()
app = Flask(__name__)

# Configurações
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

# Configuração do Tesseract (ajuste para seu sistema)
# NOTA: Se você usa Windows descomente a próxima linha e coloque o caminho apropriado
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\teste'

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def pdf_to_png(filepath):
    """Converte PDF para PNG usando PyMuPDF"""
    try:
        doc = fitz.open(filepath)
        output_paths = []
        
        for i, page in enumerate(doc):
            # Converter página para imagem (300 DPI para melhor qualidade)
            pix = page.get_pixmap(dpi=300)
            
            # Nome do arquivo de saída
            output_filename = f"{os.path.splitext(os.path.basename(filepath))[0]}_page_{i+1}.png"
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
            
            # Salvar a imagem
            pix.save(output_path)
            output_paths.append(output_path)
        
        return output_paths[0] if output_paths else None  # Retorna apenas a primeira página
    except Exception as e:
        print(f"Erro ao converter PDF: {e}")
        return None

def extract_text(image_path):
    """Extrai texto de uma imagem usando Tesseract OCR"""
    try:
        # Ler a imagem
        img = cv2.imread(image_path)
        if img is None:
            return ""
        
        # Pré-processamento básico
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        
        # Configuração do OCR (português + inglês)
        config = '--psm 6 --oem 3 -l por+eng'
        text = pytesseract.image_to_string(thresh, config=config)
        
        return text.strip()
    except Exception as e:
        print(f"Erro no OCR: {e}")
        return ""

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'Nenhum arquivo enviado'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Nenhum arquivo selecionado'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Tipo de arquivo não permitido'}), 400
    
    # Salvar o arquivo
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    file.save(filepath)
    
    # Processar PDF ou imagem
    if filename.lower().endswith('.pdf'):
        image_path = pdf_to_png(filepath)
        if not image_path:
            return jsonify({'error': 'Falha ao converter PDF'}), 500
    else:
        image_path = filepath
    
    # Extrair texto
    extracted_text = extract_text(image_path)

    result = client.responses.create(
        model="gpt-4o",
        
        input=GPT_CONTEXT + f'"{extracted_text}"'
    )
    
    return jsonify({
        'success': True,
        'text': result.output_text,
        'image': os.path.basename(image_path) if image_path else None
    })

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
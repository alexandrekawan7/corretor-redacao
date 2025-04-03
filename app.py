# Carrega variáveis de ambiente do arquivo .env (contendo a chave da API da OpenAI)
from dotenv import load_dotenv
load_dotenv()

# Importações de bibliotecas necessárias
from flask import Flask, render_template, request, jsonify  # Para criar a aplicação web
import fitz  # PyMuPDF - para trabalhar com arquivos PDF
import cv2  # OpenCV - processamento de imagens
import pytesseract  # Tesseract OCR - reconhecimento de texto em imagens
import os  # Para operações com sistema de arquivos
from openai import OpenAI  # Cliente oficial da OpenAI
from werkzeug.utils import secure_filename  # Para sanitizar nomes de arquivos


# Instruções detalhadas para o modelo GPT sobre como corrigir a redação
GPT_CONTEXT = """
    "Analise esta redação atribuindo notas de 0 a 200 para cada uma das 5 competências do ENEM. Formate a resposta EXATAMENTE como no modelo abaixo, sem comentários adicionais:

Nota Competência 1 (Domínio formal): [nota]/200  
Nota Competência 2 (Tema): [nota]/200  
Nota Competência 3 (Argumentação): [nota]/200  
Nota Competência 4 (Coesão): [nota]/200  
Nota Competência 5 (Intervenção): [nota]/200  
Total: [total]/1000

Erros ortográficos/gramaticais (sublinhados):  
[Texto da redação com os erros marcados 'O̲l̲á̲ ̲c̲a̲d̲e̲i̲a̲ ̲d̲e̲ ̲t̲e̲x̲t̲o̲,̲ ̲m̲i̲n̲h̲a̲ ̲v̲e̲l̲h̲a̲ ̲a̲m̲i̲g̲a̲' desta forma]  

Aplique rigorosamente estes critérios:  
1. Sublinhe apenas erros objetivos (ortografia, acentuação, concordância)  
2. Não justifique as notas  
3. Mantenha o formato solicitado  
4. Preserve o conteúdo original mesmo com erros  

Redação para análise: 
"""

# Inicializa o cliente da OpenAI usando a chave da API do .env
client = OpenAI()

# Cria a aplicação Flask
app = Flask(__name__)

# Configurações da aplicação
UPLOAD_FOLDER = 'static/uploads'  # Pasta onde os uploads serão salvos
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg'}  # Extensões de arquivo permitidas
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limite de 16MB para uploads

# Configuração opcional do caminho do Tesseract para Windows
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def allowed_file(filename):
    """Verifica se a extensão do arquivo está na lista de permitidas"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def pdf_to_png(filepath):
    """Converte um arquivo PDF em uma imagem PNG (apenas primeira página)"""
    try:
        doc = fitz.open(filepath)  # Abre o PDF
        output_paths = []
        
        for i, page in enumerate(doc):
            # Cria uma imagem de alta qualidade (300 DPI) da página
            pix = page.get_pixmap(dpi=300)
            
            # Gera um nome seguro para o arquivo de saída
            output_filename = f"{os.path.splitext(os.path.basename(filepath))[0]}_page_{i+1}.png"
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
            
            # Salva a imagem como PNG
            pix.save(output_path)
            output_paths.append(output_path)
        
        # Retorna apenas a primeira página convertida
        return output_paths[0] if output_paths else None
    except Exception as e:
        print(f"Erro ao converter PDF: {e}")
        return None

def extract_text(image_path):
    """Extrai texto de uma imagem usando OCR"""
    try:
        # Carrega a imagem usando OpenCV
        img = cv2.imread(image_path)
        if img is None:
            return ""
        
        # Pré-processamento da imagem para melhorar a precisão do OCR
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Converte para tons de cinza
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)  # Binariza a imagem
        
        # Configura o Tesseract para português com modo de página adequado
        config = '--psm 6 --oem 3 -l por'
        text = pytesseract.image_to_string(thresh, config=config)
        
        return text.strip()
    except Exception as e:
        print(f"Erro no OCR: {e}")
        return ""

# Rota principal que exibe o formulário de upload
@app.route('/')
def home():
    return render_template('index.html')

# Rota que processa o upload da redação e retorna a correção
@app.route('/upload', methods=['POST'])
def upload_file():
    # Verifica se o arquivo foi enviado corretamente
    if 'file' not in request.files:
        return jsonify({'error': 'Nenhum arquivo enviado'}), 400
    
    file = request.files['file']
    
    # Verifica se um arquivo foi selecionado
    if file.filename == '':
        return jsonify({'error': 'Nenhum arquivo selecionado'}), 400
    
    # Verifica se a extensão é permitida
    if not allowed_file(file.filename):
        return jsonify({'error': 'Tipo de arquivo não permitido'}), 400
    
    # Cria um nome seguro para o arquivo e salva no servidor
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    file.save(filepath)
    
    # Converte PDF para PNG se necessário
    if filename.lower().endswith('.pdf'):
        image_path = pdf_to_png(filepath)
        if not image_path:
            return jsonify({'error': 'Falha ao converter PDF'}), 500
    else:
        image_path = filepath
    
    # Extrai o texto da imagem usando OCR
    extracted_text = extract_text(image_path)

    # Envia o texto extraído para o GPT-4 com as instruções de correção
    result = client.responses.create(
        model="gpt-4o",
        input=GPT_CONTEXT + f'"{extracted_text}"'
    )
    
    # Retorna o resultado da correção em formato JSON
    return jsonify({
        'success': True,
        'text': result.output_text,
        'image': os.path.basename(image_path) if image_path else None
    })

# Ponto de entrada principal
if __name__ == '__main__':
    # Garante que a pasta de uploads existe
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    # Inicia o servidor Flask em modo de desenvolvimento
    app.run(debug=True)
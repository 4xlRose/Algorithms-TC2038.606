from flask import Flask, render_template, request, jsonify, session
import os
from werkzeug.utils import secure_filename

# Implementacion de algoritmos en una aplicacion web
# Utilizando Flask
# Por Pedro Sotelo Arce y Axel Ariel Grande Ruiz
# Ultima modificacion 17/10/2024

app = Flask(__name__)
app.secret_key = 'your_secret_key'

app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
ALLOWED_EXTENSIONS = {'txt'}

# Funcion para verificar si es valido el archivo
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Funcion para leer archivos
def leer_archivo(archivo):
    with open(archivo, 'r', encoding='utf-8') as f:
        return f.read()


#region Algoritmo Z
# Implementacion 
# Busqueda de patrones
# Por: Pedro Sotelo Arce 
def Z_algorithm(text, pattern):
        concatenated = pattern + '$' + text
        Z = [0] * len(concatenated)
        left, right, K = 0, 0, 0
        pattern_length = len(pattern)
        result_positions = []

        for i in range(1, len(concatenated)):
            if i > right:
                left, right = i, i
                while right < len(concatenated) and concatenated[right] == concatenated[right - left]:
                    right += 1
                Z[i] = right - left
                right -= 1
            else:
                K = i - left
                if Z[K] < right - i + 1:
                    Z[i] = Z[K]
                else:
                    left = i
                    while right < len(concatenated) and concatenated[right] == concatenated[right - left]:
                        right += 1
                    Z[i] = right - left
                    right -= 1
            if Z[i] == pattern_length:
                result_positions.append(i - pattern_length - 1)  # Guardar posición

        return result_positions
#endregion


#region Manacher

# Algoritmo Manacher
# Busqueda de palindromos
# Por: Pedro Sotelo Arce 

# Funcion para limpiar el texto y manacher no se muera
def limpiar_texto(text):
    # Eliminar espacios en blanco al inicio y al final
    text = text.strip()
    # Reemplazar múltiples espacios con un solo espacio
    text = ' '.join(text.split())
    return text

# Implementacion de manacher
def manacher(s):
    n = len(s)
    if n == 0:
        return -1, 0

    T = "@#" + "#".join(s) + "#$"
    nT = len(T)
    P = [0] * nT
    center = right = 0
    max_len = 0
    start = 0

    for i in range(1, nT - 1):
        mirror = 2 * center - i
        if i < right:
            P[i] = min(right - i, P[mirror])

        while (i + 1 + P[i] < nT - 1 and
               i - 1 - P[i] >= 0 and
               T[i + 1 + P[i]] == T[i - 1 - P[i]]):
            P[i] += 1
        
        if i + P[i] > right:
            center = i
            right = i + P[i]

        if P[i] > max_len:
            max_len = P[i]
            start = (i - max_len) // 2

    return start, max_len
#endregion

#region LCS
# Algoritmo LCS
# Por: Axel Ariel Grande Ruiz
def LCS(text1, text2):
    m, n = len(text1), len(text2)
    longest = 0  
    end_index = 0  
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:  
                dp[i][j] = dp[i - 1][j - 1] + 1  
                if dp[i][j] > longest:
                    longest = dp[i][j]  
                    end_index = i  
            else:
                dp[i][j] = 0  

    start_index = end_index - longest
    return text1[start_index:end_index]  
#endregion


#region Tries
# Tries
# Por: Axel Ariel Grande Ruiz
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True

    def insert_text(self, text: str):
        for word in text.split():
            self.insert(word.lower())

    def autocomplete(self, prefix: str):
        node = self.search_prefix(prefix)
        if not node:
            return []

        suggestions = []

        def dfs(current_node, path):
            if current_node.is_end_of_word:
                suggestions.append("".join(path))
            for char, next_node in current_node.children.items():
                dfs(next_node, path + [char])

        dfs(node, list(prefix))
        return suggestions

    def search_prefix(self, prefix: str):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return None
            node = node.children[char]
        return node

trie = Trie()
#endregion

#region Servidor
# Conexion a aplicacion (Rutas)
@app.route('/')
def index():
    return render_template('resultado.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file1' not in request.files or 'file2' not in request.files:
        return 'Ambos archivos deben ser enviados'

    file1 = request.files['file1']
    file2 = request.files['file2']

    if file1.filename == '' or file2.filename == '':
        return 'Ambos archivos deben ser seleccionados'

    if file1 and allowed_file(file1.filename) and file2 and allowed_file(file2.filename):
        # Guardar los archivos con nombres seguros
        filename1 = secure_filename(file1.filename)
        filename2 = secure_filename(file2.filename)

        # Guardar los archivos en la carpeta de subida
        filepath1 = os.path.join(app.config['UPLOAD_FOLDER'], filename1)
        filepath2 = os.path.join(app.config['UPLOAD_FOLDER'], filename2)
        file1.save(filepath1)
        file2.save(filepath2)

        # Almacenar los nombres de archivo en la sesión
        session['file1'] = filename1
        session['file2'] = filename2

        # Leer el contenido de los archivos
        text1 = leer_archivo(filepath1)
        text2 = leer_archivo(filepath2)
        
        trie.insert_text(text1)


        return render_template('resultado.html', texto1=text1, texto2=text2)

    return 'Alguno de los archivos no es válido'

# Ruta especifica para el Algoritmo Z
@app.route('/z_search', methods=['POST'])
def z_search():
    patron = request.form.get('patron')
    
    # Acceder a los nombres de los archivos desde la sesión
    filename1 = session.get('file1')
    filename2 = session.get('file2')

    if not filename1 or not filename2:
        return 'Uno de los archivos no ha sido cargado'

    filepath1 = os.path.join(app.config['UPLOAD_FOLDER'], filename1)
    filepath2 = os.path.join(app.config['UPLOAD_FOLDER'], filename2)

    text1 = leer_archivo(filepath1)
    text2 = leer_archivo(filepath2)

    positions = Z_algorithm(text1, patron)

    # Resaltar las coincidencias en el texto1
    highlighted_text1 = ""
    last_index = 0
    
    # Resaltar las coincidencias en el texto1
    for pos in positions:
        # Solo resaltar si es una coincidencia exacta
        if text1[pos:pos + len(patron)] == patron:
            highlighted_text1 += text1[last_index:pos] + f'<span class="highlight">{text1[pos:pos + len(patron)]}</span>'
            last_index = pos + len(patron)

    # Agregar cualquier texto restante después de la última coincidencia
    highlighted_text1 += text1[last_index:]

    return render_template('resultado.html', texto1=highlighted_text1, texto2=text2)

# Ruta especifica para el algoritmo Manacher
@app.route('/manacher_search', methods=['POST'])
def manacher_search():
    # Acceder a los nombres de los archivos desde la sesión
    filename1 = session.get('file1')
    filename2 = session.get('file2')

    if not filename1 or not filename2:
        return 'Uno de los archivos no ha sido cargado'

    filepath1 = os.path.join(app.config['UPLOAD_FOLDER'], filename1)
    filepath2 = os.path.join(app.config['UPLOAD_FOLDER'], filename2)

    text1 = limpiar_texto(leer_archivo(filepath1))
    text2 = limpiar_texto(leer_archivo(filepath2))

    start, length = manacher(text1)
    if length > 0:
        text1 = text1[:start] + f'<span class="highlight2">{text1[start:start + length]}</span>' + text1[start + length:]

    return render_template('resultado.html', texto1=text1, texto2=text2)

# Ruta especifica para el LCS
@app.route('/lcs', methods=['POST'])
def lcs_route():
    filename1 = session.get('file1')
    filename2 = session.get('file2')

    if not filename1 or not filename2:
        return 'Uno de los archivos no ha sido cargado'

    filepath1 = os.path.join(app.config['UPLOAD_FOLDER'], filename1)
    filepath2 = os.path.join(app.config['UPLOAD_FOLDER'], filename2)

    text1 = leer_archivo(filepath1)
    text2 = leer_archivo(filepath2)

    lcs_result = LCS(text1, text2)

    if lcs_result != "": 
        highlighted_lcs = f'<span style="background-color: #add8e6;">{lcs_result}</span>'
        texto1_resaltado = text1.replace(lcs_result, highlighted_lcs)
        texto2_resaltado = text2.replace(lcs_result, highlighted_lcs)

    return render_template(
        'resultado.html',
        lcs_result=lcs_result, texto1=texto1_resaltado, texto2=texto2_resaltado)

# Ruta especifica para el Trie
@app.route('/autocomplete', methods=['GET'])
def autocomplete():
    prefix = request.args.get('prefix', '').lower()
    suggestions = trie.autocomplete(prefix)
    return jsonify({"suggestions": suggestions})

if __name__ == '__main__':
    app.run(debug=True)

#endregion

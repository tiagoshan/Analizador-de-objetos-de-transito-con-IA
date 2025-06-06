import os
import traceback
from flask import Flask, request, render_template, jsonify, flash, redirect, url_for
import tensorflow as tf
from PIL import Image
import numpy as np
from werkzeug.utils import secure_filename
import logging

# Configurar logging más detallado
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Función para cargar el modelo con más información de debug
def load_model_with_debug():
    model_path = 'my_traffic_object_model.h5'
    
    # Verificar si el archivo existe
    if not os.path.exists(model_path):
        logger.error(f"❌ El archivo del modelo no existe: {model_path}")
        logger.info(f"📁 Archivos en el directorio actual: {os.listdir('.')}")
        return None
    
    try:
        logger.info(f"🔄 Cargando modelo desde: {model_path}")
        model = tf.keras.models.load_model(model_path)
        logger.info(f"✅ Modelo cargado exitosamente")
        
        # Información del modelo
        logger.info(f"📊 Información del modelo:")
        logger.info(f"   - Input shape: {model.input_shape}")
        logger.info(f"   - Output shape: {model.output_shape}")
        logger.info(f"   - Número de capas: {len(model.layers)}")
        
        # Probar predicción con datos dummy
        try:
            dummy_input = np.random.random((1, 128, 128, 3))
            dummy_prediction = model.predict(dummy_input, verbose=0)
            logger.info(f"🧪 Prueba con datos dummy exitosa. Shape de salida: {dummy_prediction.shape}")
        except Exception as e:
            logger.error(f"❌ Error en prueba con datos dummy: {e}")
        
        return model
        
    except Exception as e:
        logger.error(f"❌ Error cargando el modelo: {e}")
        logger.error(f"📋 Traceback completo: {traceback.format_exc()}")
        return None

# Cargar el modelo
model = load_model_with_debug()

# Inicializar Flask
app = Flask(__name__)
app.secret_key = 'debug_key_12345'

# Configuración
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
MAX_FILE_SIZE = 16 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Crear carpeta de uploads
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
    logger.info(f"📁 Carpeta {UPLOAD_FOLDER} creada")

# Mapeo de clases - AJUSTAR SEGÚN TU MODELO
CLASS_NAMES = {
    0: "Bicicleta",
    1: "Carro", 
    2: "Límite de velocidad",
    3: "Persona",
    4: "Carro",
    5: "Semaforo",
    6: "Camion"
    # Agregar más según tu modelo
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image_debug(image_path, target_size=(128, 128)):
    """Preprocesar imagen con debug detallado"""
    try:
        logger.info(f"🖼️ Procesando imagen: {image_path}")
        
        # Verificar que el archivo existe
        if not os.path.exists(image_path):
            logger.error(f"❌ El archivo de imagen no existe: {image_path}")
            return None
        
        # Obtener información del archivo
        file_size = os.path.getsize(image_path)
        logger.info(f"📏 Tamaño del archivo: {file_size} bytes")
        
        # Abrir imagen
        logger.info(f"🔄 Abriendo imagen...")
        image = Image.open(image_path)
        logger.info(f"✅ Imagen abierta exitosamente")
        logger.info(f"   - Formato original: {image.format}")
        logger.info(f"   - Modo: {image.mode}")
        logger.info(f"   - Tamaño: {image.size}")
        
        # Convertir a RGB si es necesario
        if image.mode != 'RGB':
            logger.info(f"🔄 Convirtiendo de {image.mode} a RGB")
            image = image.convert('RGB')
        
        # Redimensionar
        logger.info(f"🔄 Redimensionando a {target_size}")
        original_size = image.size
        image = image.resize(target_size)
        logger.info(f"✅ Redimensionado de {original_size} a {image.size}")
        
        # Convertir a array numpy
        logger.info(f"🔄 Convirtiendo a array numpy")
        image_array = np.array(image)
        logger.info(f"✅ Array creado. Shape: {image_array.shape}, dtype: {image_array.dtype}")
        logger.info(f"   - Min value: {image_array.min()}, Max value: {image_array.max()}")
        
        # Normalizar
        logger.info(f"🔄 Normalizando valores")
        image_array = image_array / 255.0
        logger.info(f"✅ Normalizado. Min: {image_array.min()}, Max: {image_array.max()}")
        
        # Expandir dimensiones
        logger.info(f"🔄 Expandiendo dimensiones para batch")
        image_array = np.expand_dims(image_array, axis=0)
        logger.info(f"✅ Shape final: {image_array.shape}")
        
        return image_array
        
    except Exception as e:
        logger.error(f"❌ Error en preprocess_image_debug: {e}")
        logger.error(f"📋 Traceback: {traceback.format_exc()}")
        return None

def predict_traffic_sign_debug(image_path):
    """Predicción con debug detallado"""
    logger.info(f"🎯 Iniciando predicción para: {image_path}")
    
    if model is None:
        logger.error(f"❌ Modelo no está cargado")
        return None, None, "Modelo no disponible"
    
    try:
        # Preprocesar imagen
        image_array = preprocess_image_debug(image_path)
        if image_array is None:
            return None, None, "Error en preprocesamiento de imagen"
        
        # Realizar predicción
        logger.info(f"🔄 Realizando predicción...")
        predictions = model.predict(image_array, verbose=1)
        logger.info(f"✅ Predicción completada")
        logger.info(f"   - Shape de predicción: {predictions.shape}")
        logger.info(f"   - Valores de predicción: {predictions[0]}")
        
        # Obtener clase predicha
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = float(np.max(predictions))
        
        logger.info(f"📊 Resultados:")
        logger.info(f"   - Clase predicha: {predicted_class}")
        logger.info(f"   - Confianza: {confidence:.4f} ({confidence*100:.2f}%)")
        logger.info(f"   - Todas las probabilidades: {predictions[0]}")
        
        return predicted_class, confidence, None
        
    except Exception as e:
        error_msg = f"Error en predicción: {e}"
        logger.error(f"❌ {error_msg}")
        logger.error(f"📋 Traceback: {traceback.format_exc()}")
        return None, None, error_msg

@app.route('/')
def home():
    return render_template('debug_index.html', model_loaded=model is not None)

@app.route('/predict', methods=['POST'])
def predict():
    logger.info(f"🚀 Iniciando predicción...")
    
    # Verificar modelo
    if model is None:
        error_msg = "Modelo no está disponible"
        logger.error(f"❌ {error_msg}")
        flash(error_msg)
        return redirect(url_for('home'))
    
    # Verificar archivo
    if 'file' not in request.files:
        error_msg = "No se encontró archivo en la petición"
        logger.error(f"❌ {error_msg}")
        flash(error_msg)
        return redirect(url_for('home'))
    
    file = request.files['file']
    logger.info(f"📁 Archivo recibido: {file.filename}")
    
    if file.filename == '':
        error_msg = "Nombre de archivo vacío"
        logger.error(f"❌ {error_msg}")
        flash(error_msg)
        return redirect(url_for('home'))
    
    # Verificar extensión
    if not allowed_file(file.filename):
        error_msg = f"Tipo de archivo no permitido: {file.filename}"
        logger.error(f"❌ {error_msg}")
        flash(error_msg)
        return redirect(url_for('home'))
    
    try:
        # Guardar archivo
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        logger.info(f"💾 Guardando archivo en: {file_path}")
        
        file.save(file_path)
        logger.info(f"✅ Archivo guardado exitosamente")
        
        # Verificar que se guardó correctamente
        if not os.path.exists(file_path):
            error_msg = "Error: archivo no se guardó correctamente"
            logger.error(f"❌ {error_msg}")
            flash(error_msg)
            return redirect(url_for('home'))
        
        saved_size = os.path.getsize(file_path)
        logger.info(f"✅ Archivo verificado. Tamaño: {saved_size} bytes")
        
        # Realizar predicción
        predicted_class, confidence, error = predict_traffic_sign_debug(file_path)
        
        # Limpiar archivo temporal
        try:
            os.remove(file_path)
            logger.info(f"🗑️ Archivo temporal eliminado")
        except:
            logger.warning(f"⚠️ No se pudo eliminar archivo temporal")
        
        if error:
            flash(f"Error en predicción: {error}")
            return redirect(url_for('home'))
        
        if predicted_class is None:
            flash("Error: no se pudo realizar la predicción")
            return redirect(url_for('home'))
        
        # Preparar resultado
        class_name = CLASS_NAMES.get(predicted_class, f"Clase {predicted_class}")
        result = {
            'class_id': int(predicted_class),
            'class_name': class_name,
            'confidence': round(confidence * 100, 2)
        }
        
        logger.info(f"🎉 Predicción exitosa: {result}")
        return render_template('debug_result.html',filename=filename, result=result)
        
    except Exception as e:
        error_msg = f"Error inesperado: {e}"
        logger.error(f"❌ {error_msg}")
        logger.error(f"📋 Traceback: {traceback.format_exc()}")
        flash(error_msg)
        return redirect(url_for('home'))

@app.route('/debug')
def debug_info():
    """Endpoint para información de debug"""
    info = {
        'model_loaded': model is not None,
        'tensorflow_version': tf.__version__,
        'uploads_folder_exists': os.path.exists(UPLOAD_FOLDER),
        'current_directory': os.getcwd(),
        'files_in_directory': os.listdir('.'),
        'python_path': os.environ.get('PYTHONPATH', 'Not set'),
    }
    
    if model is not None:
        info['model_input_shape'] = str(model.input_shape)
        info['model_output_shape'] = str(model.output_shape)
    
    return jsonify(info)

@app.route('/test-prediction')
def test_prediction():
    """Endpoint para probar predicción con imagen dummy"""
    if model is None:
        return jsonify({'error': 'Modelo no cargado'})
    
    try:
        # Crear imagen dummy
        dummy_image = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        dummy_image_pil = Image.fromarray(dummy_image)
        
        # Guardar temporalmente
        test_path = os.path.join(UPLOAD_FOLDER, 'test_dummy.png')
        dummy_image_pil.save(test_path)
        
        # Probar predicción
        predicted_class, confidence, error = predict_traffic_sign_debug(test_path)
        
        # Limpiar
        os.remove(test_path)
        
        if error:
            return jsonify({'error': error})
        
        return jsonify({
            'success': True,
            'predicted_class': int(predicted_class),
            'confidence': float(confidence)
        })
        
    except Exception as e:
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()})

if __name__ == '__main__':
    logger.info(f"🚀 Iniciando aplicación Flask en modo debug")
    logger.info(f"📊 Estado del modelo: {'✅ Cargado' if model else '❌ No cargado'}")
    app.run(debug=True, host='0.0.0.0', port=5000)
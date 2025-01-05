import pickle
from flask import Flask, request, render_template
import regex as re
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

app = Flask(__name__)

# Load the model
try:
    model = load_model('/Users/amnaalmurshda/Desktop/FLASK_Trans/models/gru_seq2seq_model.h5')
    print("GRU-based Seq2Seq model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Load the tokenizers
try:
    with open('/Users/amnaalmurshda/Desktop/FLASK_Trans/models/eng_tokenizer.pkl', 'rb') as file:
        eng_tokenizer = pickle.load(file)
    with open('/Users/amnaalmurshda/Desktop/FLASK_Trans/models/fr_tokenizer.pkl', 'rb') as file:
        fr_tokenizer = pickle.load(file)
    print("Tokenizers loaded successfully.")
except Exception as e:
    print(f"Error loading tokenizers: {e}")
    eng_tokenizer = None
    fr_tokenizer = None

def preprocess_text(text):
    text = re.sub(r'[.!?:;,]', '', text)
    return text

@app.route('/')
def index():
    print("Rendering index.html")
    return render_template('index.html', original="", translated="")

@app.route('/translate', methods=['POST'])
def translate():
    if model is None or eng_tokenizer is None or fr_tokenizer is None:
        return render_template('index.html', original="", translated="Error: Model or tokenizers not loaded.")
    
    text = request.form['text']
    try:
        # Preprocess and tokenize the input text
        preprocessed_text = preprocess_text(text)
        sequence = eng_tokenizer.texts_to_sequences([preprocessed_text])
        padded_sequence = pad_sequences(sequence, maxlen=model.input_shape[1], padding='post')
        
        # Use the model to translate the text
        prediction = model.predict(padded_sequence)
        y_id_to_word = {value: key for key, value in fr_tokenizer.word_index.items()}
        y_id_to_word[0] = '<PAD>'
        translated_text = ' '.join([y_id_to_word[np.argmax(x)] for x in prediction[0]])
        print(f"Translation successful: {translated_text}")
    except Exception as e:
        translated_text = f"Error during translation: {e}"
        print(f"Error during translation: {e}")
    
    return render_template('index.html', original=text, translated=translated_text)

if __name__ == '__main__':
    app.run(debug=True, port=9930)

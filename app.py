from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
import os

app = Flask(__name__)

class PoetryGenerator:
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.max_sequence_length = None
        self.load_saved_model()

    def load_saved_model(self, model_dir='model'):
        """Load saved model assets"""
        try:
            model_path = os.path.join(model_dir, 'poetry_model.h5')
            if not os.path.exists(model_path):
                model_path = os.path.join(model_dir, 'poetry_model.keras')
            self.model = load_model(model_path)
            
            with open(os.path.join(model_dir, 'tokenizer.pkl'), 'rb') as f:
                self.tokenizer = pickle.load(f)
                
            self.max_sequence_length = self.model.input_shape[1] + 1
            
            print("Model loaded successfully!")
            print(f"Max sequence length: {self.max_sequence_length}")
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def generate_line(self, seed_text, next_words, temperature=1.0):
        """Generate a single line of poetry"""
        for _ in range(next_words):
            token_list = self.tokenizer.texts_to_sequences([seed_text])[0]
            token_list = pad_sequences(
                [token_list],
                maxlen=self.max_sequence_length-1,
                padding='pre'
            )
            predictions = self.model.predict(token_list, verbose=0)[0]
            predictions = self._apply_temperature(predictions, temperature)
            predicted_index = np.random.choice(len(predictions), p=predictions)
            
            output_word = ""
            for word, index in self.tokenizer.word_index.items():
                if index == predicted_index:
                    output_word = word
                    break
                    
            seed_text += " " + output_word
        return seed_text

    def _apply_temperature(self, predictions, temperature):
        """Apply temperature to prediction probabilities"""
        predictions = np.asarray(predictions).astype('float64')
        predictions = np.log(predictions) / temperature
        exp_preds = np.exp(predictions)
        return exp_preds / np.sum(exp_preds)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_poetry():
    data = request.get_json()
    seed_text = data.get('text', '') 
    num_lines = int(data.get('num_lines', 1)) 
    words_per_line = int(data.get('words_per_line', 1)) 
    temperature = float(data.get('temperature', 1.0))
    
    if not seed_text:
        return jsonify({'error': 'Please enter some starting words'}), 400
    
    try:

        line = generator.generate_line(seed_text, words_per_line, temperature)
        next_word = line[len(seed_text):].strip()
        
        return jsonify({
            'next_word': next_word,
            'full_line': line
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    generator = PoetryGenerator()
    app.run(debug=True)
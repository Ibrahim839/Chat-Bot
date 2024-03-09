from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import json

with open('API/data1.json') as file:
    data=json.load(file)

app = Flask(__name__)

@app.route('/', methods=['GET'])


def index():
    return jsonify({
        'status':'healthy'
    })
# Load the model and tokenizer
model = load_model('model.h5')
tokenizer = pickle.load(open("API/toknizer.pkl", "rb"))
label_encoder = pickle.load(open("API/label_encoder.pkl", "rb"))


def predict_response(input_text):
    sent = pad_sequences(tokenizer.texts_to_sequences([input_text]), maxlen=20)
    result = np.argmax(model.predict(np.array(sent), verbose=0))
    f_res = label_encoder.inverse_transform(np.array(result).reshape(1))

    for label in data['data']:
        if label['label'] == f_res:
            return np.random.choice(label['responses'])

# Now you can use this function within your Flask API endpoint
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_text = data['text']
    response = predict_response(input_text)
    return jsonify({'response': response})



if __name__=='__main__':
    
    app.run(
        host= '127.0.0.1',
        port= 5000,
        debug= True
    )
    

#if __name__ == '__main__':
#    app.run(debug=True)

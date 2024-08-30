import joblib
import pandas as pd
import sys
import json
import numpy as np
from flask import Flask, jsonify, request
import logging

app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
def call_home():
    print(request.values)
    return "SERVER IS RUNNING!"

def init():
    """
    Esta função é chamada quando o contêiner é inicializado/iniciado, normalmente após a criação/atualização da implantação.
    Você pode escrever a lógica aqui para realizar operações de inicialização, como armazenar o modelo em cache na memória
    """
    global model
    # desserializa o arquivo do modelo de volta em um modelo sklearn
    model = joblib.load('model.pkl')
    logging.info("Init complete")

@app.route("/score", methods=['POST'])
def run():
    """
    Esta função é chamada para cada chamada para executar a predição real.
    No exemplo, extraímos os dados da entrada json e chamamos predict() do modelo scikit-learn
    E retornamos o resultado em formato de lista.
    """
    try:
        logging.info("model 1: request received")
        raw_data = request.data
        data = json.loads(raw_data)["data"]
        data = np.array(data)
        result = model.predict(data)
        logging.info("Request processed")

        return jsonify(result.tolist())
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    init()
    app.run(port=80, host='0.0.0.0')
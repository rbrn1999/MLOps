from flask import Flask, request, json
import subprocess

app = Flask(__name__)

@app.route('/')
def hello():
    return 'Webhooks with Python'

@app.route('/commit_webhook', methods=['POST'])
def didCommit():
    data = request.json
    app.logger.info(data)
    subprocess.run(["python", "esgBERTv4.py"], cwd="/workspace/Step3")
    return data

if __name__ == '__main__':
    app.run(debug=True)

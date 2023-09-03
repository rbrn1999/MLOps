from flask import Flask, request
import subprocess

app = Flask(__name__)

@app.route('/')
def hello():
    return 'Webhooks with Python'

@app.route('/commit_webhook', methods=['POST'])
def didCommit():
    data = request.json
    app.logger.info(f'commit: {data["repo"]["headSha"]}')
    cur_headSha = data["repo"]["headSha"]
    prev_headSha = ""
    try:
        with open("commit.txt", "r") as f:
            prev_headSha = f.read()
    except IOError:
        app.logger.warn("can't read 'commit.txt'")

    if cur_headSha == prev_headSha:
        app.logger.info("Not a new commit, will not proceed to training.")
        return "Not a new commit, will not proceed to training."

    with open("commit.txt", "w+") as f:
        f.write(cur_headSha)
    subprocess.run(["python", "main.py"], cwd="/workspace/MLOps")

    return data

if __name__ == '__main__':
    app.run('0.0.0.0', debug=True)

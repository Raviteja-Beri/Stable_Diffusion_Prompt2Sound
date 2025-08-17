# app.py
from flask import Flask, request, send_file, render_template
from inference import generate_audio
import os

app = Flask(__name__, static_folder="static")

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json()
    prompt = data.get("prompt", "")
    file_path = generate_audio(prompt)
    return send_file(file_path, mimetype="audio/wav", as_attachment=False)

if __name__ == "__main__":
    app.run(debug=True)


from flask import Flask, request, jsonify
from flask_cors import CORS
import subprocess

app = Flask(__name__)
CORS(app)

MODEL_NAME = "llama3:8b"

@app.route("/api/chatbot", methods=["POST"])
def generate():
    data = request.get_json()
    prompt = data.get("prompt", "")
    use_json = data.get("use_json", False)  # New parameter

    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    try:
        if use_json:
            # For product generation - use JSON format
            result = subprocess.run(
                ["ollama", "run", MODEL_NAME, "--format", "json", prompt],
                capture_output=True,
                text=True,
                check=True
            )
        else:
            # For category validation - use plain text
            result = subprocess.run(
                ["ollama", "run", MODEL_NAME, prompt],
                capture_output=True,
                text=True,
                check=True
            )
        
        output_text = result.stdout.strip()
        return jsonify({"reply": output_text})

    except subprocess.CalledProcessError as e:
        return jsonify({"error": e.stderr}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
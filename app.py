from flask import Flask, request, jsonify, render_template, send_from_directory
import os, random
from werkzeug.utils import secure_filename
from gtts import gTTS
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.models import Model
import pickle
from flask_cors import CORS
from tensorflow.keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)
UPLOAD_FOLDER = "uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load Models and Tokenizer
try:
    caption_model = load_model("model.keras")
    feature_extractor = load_model("feature_extractor.keras")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
except FileNotFoundError as e:
    print(f"Error loading model or tokenizer: {e}")
    caption_model = None
    feature_extractor = None
    tokenizer = None

max_length = 34
img_size = 224

def generate_and_display_caption(image_path):
    if caption_model is None or feature_extractor is None or tokenizer is None:
        return "Model or tokenizer not loaded."

    try:
        img = load_img(image_path, target_size=(img_size, img_size))
        img = img_to_array(img) / 255.0
        img = np.expand_dims(img, axis=0)
        image_features = feature_extractor.predict(img, verbose=0)

        in_text = "startseq"
        for i in range(max_length):
            sequence = tokenizer.texts_to_sequences([in_text])[0]
            sequence = pad_sequences([sequence], maxlen=max_length)
            yhat = caption_model.predict([image_features, sequence], verbose=0)
            yhat_index = np.argmax(yhat)
            word = tokenizer.index_word.get(yhat_index, None)
            if word is None or word == "endseq":
                break
            in_text += " " + word
        caption = in_text.replace("startseq", "").replace("endseq", "").strip()
        return caption
    except FileNotFoundError:
        return "Image file not found."
    except Exception as e:
        return f"Error generating caption: {e}"

def text_to_speech(text):
    try:
        if os.path.exists("static/output.mp3"):
            os.remove("static/output.mp3")
    except Exception as a:
        print(a)
    tts = gTTS(text=text, lang='en')
    tts.save("static/output.mp3")
    return "static/output.mp3"

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory("static", filename)

@app.route("/generate_caption", methods=["POST"])
def caption_image():
    if caption_model is None:
        return jsonify({"error": "Model not loaded"}), 500

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)

    # try:
    #     if os.path.exists(audio_directory):
    #         os.remove(audio_directory) 
    # except Exception as e:
    #     pass
    try:
        file.save(file_path) 
        caption_new = generate_and_display_caption(file_path)
        return jsonify({"caption": caption_new})
    except Exception as e:
        return jsonify({"error": f"Error processing image: {e}"}), 500

@app.route("/generate_audio", methods=["POST"])
def caption_audio():
    data = request.get_json()
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400

    audio_directory = text_to_speech(text)
    return jsonify({"audio_url": audio_directory})

# get api to return audio object to play directly in browser
@app.route("/get_audio", methods=["GET"])
def get_audio():
    audio_path = "static/output.mp3"
    if os.path.exists(audio_path):
        return send_from_directory("static", "output.mp3")
    else:
        return jsonify({"error": "Audio file not found"}), 404


if __name__ == "__main__":
    app.run(debug=True)
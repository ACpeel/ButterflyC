import os
from flask import (
    Flask,
    request,
    jsonify,
    render_template,
    redirect,
    url_for,
    session,
    send_from_directory,
)
import logging
import secrets
from urllib.parse import quote

from werkzeug.utils import secure_filename

from main.recognize import recognize
from main.utils.config import load_config


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

configs = load_config()
static = os.path.abspath(configs["static"])
uploaded_folder = os.path.abspath(configs["upload_dir"])
os.makedirs(uploaded_folder, exist_ok=True)

app = Flask(__name__, template_folder="templates", static_folder=None)
secret_key = secrets.token_hex(24)
with open("key.yml", "w", encoding="utf-8") as key_w:
    key_w.write(f'Web_Secret_Key: "{secret_key}"\n')
app.secret_key = secret_key


def clear_uploaded_folder():
    for filename in os.listdir(uploaded_folder):
        file_path = os.path.join(uploaded_folder, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)


def build_upload_path(filename):
    safe_name = secure_filename(filename) or "upload.jpg"
    generated_name = f"{secrets.token_hex(8)}_{safe_name}"
    return generated_name, os.path.join(uploaded_folder, generated_name)

@app.route("/", methods=["GET"])
def index():
    clear_uploaded_folder()
    return render_template("index.html")


@app.route("/ur", methods=['POST'])
def upload_recognize():
    if "file" not in request.files:
        return jsonify({"error": "no file part"}), 400

    pic = request.files["file"]

    if pic.filename == "":
        return jsonify({"error": "no selected file"}), 400

    if pic:
        try:
            stored_name, file_path = build_upload_path(pic.filename)
            pic.save(file_path)

            result = recognize(file_path)
            category = str(result[1][0])
            encoded_category_name = quote(
                category.split(" ")[1] if len(category.split(" ")) > 1 else category
            )

            session["result_image"] = stored_name
            session["result_category"] = category
            session["result_encoded_category"] = encoded_category_name
            return jsonify({"redirect": url_for("result")})

        except Exception as e:
            logger.exception("Recognition failed")
            return jsonify({"error": str(e)}), 500


@app.route("/result", methods=["GET"])
def result():
    image_filename = session.get("result_image")
    category = session.get("result_category")
    encoded_category = session.get("result_encoded_category")

    if not image_filename or not category:
        return redirect(url_for("index"))

    # 手动构造图片的 URL
    image_url = url_for("serve_uploaded", filename=image_filename)

    return render_template(
        "result.html",
        image=image_url,
        category=category,
        encoded_category_name=encoded_category,
    )


@app.route("/uploaded/<filename>")
def serve_uploaded(filename):
    return send_from_directory(uploaded_folder, filename)

@app.route("/static/<path:filename>")
def serve_static(filename):
    return send_from_directory(static, filename)

@app.route("/butterfly", methods=["GET"])
def butterfly():
    """
    显示蝴蝶介绍
    """
    url = "https://www.inaturalist.org/taxa/47224-Papilionoidea"
    return redirect(url)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8090, debug=True)

from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from Hispatologic_cancer_detection.configs.confs import *
import os

app_params = load_conf("configs/app_params.yml")

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = app_params["upload_folder_path"]
app.config["SECRET_KEY"] = app_params["secret_key"]


@app.route("/", methods=["GET", "POST"])
def home_page():
    if request.method == "POST":
        if request.form.get("action1") == "IMPORT IMAGE":
            return render_template("image_uploading.html")
        elif request.form.get("action2") == "LAUNCH MODEL PIPELINE":
            pass  # do something else
        else:
            pass  # unknown
    elif request.method == "GET":
        return render_template("home.html", form=request.form)
    return render_template("home.html")


@app.route("/upload")
def image_visualization():
    return render_template("image_uploading.html")


@app.route("/upload", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        while len(os.listdir(app.config["UPLOAD_FOLDER"])) > 0:
            for file in os.listdir(app.config["UPLOAD_FOLDER"]):
                os.remove(os.path.join(app.config["UPLOAD_FOLDER"], file))
        f = request.files["file"]
        f.save(os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(f.filename)))
    return redirect(url_for("show_image"))


@app.route("/image_display", methods=["GET", "POST"])
def show_image():
    print("ICIIIII", os.getcwd())
    full_filename = os.path.join(
        app.config["UPLOAD_FOLDER"], os.listdir(app.config["UPLOAD_FOLDER"])[0]
    )
    return render_template("image_display.html", user_image=full_filename)


if __name__ == "__main__":
    app.run(debug=True)

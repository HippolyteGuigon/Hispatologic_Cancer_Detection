from flask import Flask

app = Flask(__name__)


@app.route('/')
def get_app():
    return 'Hispathologic Cancer Detection Application'
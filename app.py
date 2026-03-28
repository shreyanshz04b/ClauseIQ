from flask import Flask
from app.routes import bp
from app.db import init_db, migrate_db
import os
from config import UPLOAD_FOLDER

app = Flask(__name__, template_folder="app/templates", static_folder="app/static")

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

init_db()
migrate_db()

app.register_blueprint(bp)

if __name__ == "__main__":
    app.run(debug=True)
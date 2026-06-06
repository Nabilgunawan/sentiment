import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["UPLOAD_FOLDER"] = "/tmp/uploads"

from app import app as flask_app

app = flask_app

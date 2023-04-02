from flask import Flask

app = Flask(__name__)

app.config['SECRET_KEY'] = '5792428bb0b13ae0c67ddfdf280ac245'
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

from script import routes
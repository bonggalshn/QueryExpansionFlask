from flask import Flask


app = Flask(__name__)
app.config['SECRET_KEY'] = 'ba3e4cdfb44a1ea16a2301cfef3e90e7'


from expansion import routes
import flask

app = flask.Flask(__name__)

@app.route('/')
def route_main_page():
    return flask.render_template('index.html')

@app.route('/static/<path:path>')
def serve_static():
    return send_from_directory('static', path)

@app.route('/submit', methods=['POST'])
def submit_value():
    print(flask.request.values)
    return flask.redirect('/')

if __name__ == '__main__':
    app.run('0.0.0.0', debug=True)

import flask
import os
import random

app = flask.Flask(__name__)

@app.route('/')
def route_main_page():
    filenames = os.listdir('static')
    filenames = [f for f in filenames if f.endswith('.mp4')]
    if len(filenames) == 0:
        raise ValueError("Error: No .mp4 files available in static/")
    filename = random.choice(filenames)
    class_name = filename.split('-')[-1].replace('.mp4', '')
    args = {
            'filename': filename,
            'class': class_name,
    }
    return flask.render_template('index.html', **args)

@app.route('/static/<path:path>')
def serve_static():
    return send_from_directory('static', path)

@app.route('/submit', methods=['POST'])
def submit_value():
    print("Submitted value:")
    for k, v in flask.request.form.items():
        print("\t{}: {}".format(k, v))
    return flask.redirect('/')

if __name__ == '__main__':
    app.run('0.0.0.0', debug=True)

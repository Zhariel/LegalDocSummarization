from flask import Flask, render_template, request

app = Flask(__name__)


@app.route('/', methods=['GET'])
def index():
    summary = 'this is a summary'
    if 'text' in request.args:
        print(request.args['text'])
        return render_template('summary.html', summary=summary)

    return render_template('index.html')


if __name__ == '__main__':
    app.run()

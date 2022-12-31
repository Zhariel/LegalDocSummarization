from flask import Flask, render_template, request
from nlp.train import train_model
from nlp.summarize import summarize_contract
import os

naive_bayes = train_model(sentences_path=os.path.join('nlp', 'sentences'), tags_path=os.path.join('nlp', 'tags'))
print(type(naive_bayes))

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    if 'text' in request.args:
        print(request.args['text'])
        summary = summarize_contract(model=naive_bayes, document=request.args['text'])
        return render_template('summary.html', summary=summary)

    return render_template('index.html')


if __name__ == '__main__':
    app.run()

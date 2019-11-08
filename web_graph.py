from flask import Flask, render_template, url_for, redirect, request


app = Flask(__name__)

@app.route("/")
def index():
    return '<img src="static/images/graph.jpg" width="800px" height="800px">'




if __name__ == "__main__":
    app.run(debug=True)
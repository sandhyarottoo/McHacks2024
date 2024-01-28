from flask import Flask, redirect, url_for, render_template, request
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index1.html")

@app.route('/theprogram/', methods=["POST", "GET"])
def theprogram():
    if request.method == "POST":
       user= request.form ["nm"]  #gives us result of name
       return redirect(url_for("user", usr=user))
    else: 
       return render_template('theprogram.html')

@app.route("/<usr>")
def user(usr):
   return f"<h1>{usr}</h1>"

if __name__ == "__main__" :
 app.run(debug=True) #detect changes and update

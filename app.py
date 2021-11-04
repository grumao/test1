from flask import Flask,render_template,request,redirect,url_for
import json
app = Flask(__name__)

@app.route('/')
def home():

    return render_template('home.html')

@app.route('/services', methods = ['GET','POST'])
def services():
    if request.method == 'POST':

        Data = {}
        NLPserv = {1:0,2:0,3:0,4:0,5:0,6:0}

        if request.form.get('service1'):
            NLPserv[1] = 1
        if request.form.get('service2'):
            NLPserv[2] = 1
        if request.form.get('service3'):
            NLPserv[3] = 1
        if request.form.get('service4'):
            NLPserv[4] = 1
        if request.form.get('service5'):
            NLPserv[5] = 1
        if request.form.get('service6'):
            NLPserv[6] = 1

        Data['user_string'] = {'services': NLPserv,'s1':request.form['s1'] }

        with open('Data.json','w') as Data_file:
            json.dump(Data,Data_file)

        return render_template('results.html', name=request.form['s1'])
    else:
        return redirect(url_for('home'))


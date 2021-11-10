from flask import Flask,render_template,request,redirect,url_for,flash
import json
app = Flask(__name__)
app.secret_key = 'workgroup595'
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

        Flag = 0
        for k in NLPserv.values():
            if k == 0:
                Flag += 1

        if Flag == 6:
            flash('Please select atleast one service to proceed')
            return redirect(url_for('home'))


        f = open('Data.json')
        data = json.load(f)

        def sentiment(textblock):
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            analyser = SentimentIntensityAnalyzer()
            service1 = analyser.polarity_scores(textblock)
            return service1

        def part_of_speech(textblock):
            import nltk
            nltk.download('punkt')
            nltk.download('averaged_perceptron_tagger')
            from nltk import pos_tag
            from nltk import word_tokenize
            text = word_tokenize(textblock)
            service2 = pos_tag(text)
            return service2

        text = data['user_string']['s1']
        output = {'Sentiment analysis':'-','Part of speech':'-',
                  'Text classification':'-','Word Cloud':'-',
                  'Topic modelling':'-','Aspect mining':'-'}
        if data['user_string']['services']['1']:
            op1 = sentiment(text)
            output['Sentiment analysis'] = op1
        if data['user_string']['services']['2']:
            op2 = part_of_speech(text)
            output['Part of speech'] = op2




        return render_template('results.html',R1= output['Sentiment analysis'],R2 = output['Part of speech'],
                               R3= output['Text classification'], R4= output['Word Cloud'],
                               R5= output['Topic modelling'], R6= output['Aspect mining'])
    else:
        return redirect(url_for('home'))


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)

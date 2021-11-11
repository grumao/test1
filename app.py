import matplotlib.pyplot as plt
from flask import Flask,render_template,request,redirect,url_for,flash
import json
import gensim
from gensim.utils import simple_preprocess
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import spacy
import gensim.corpora as corpora
from gensim.models import CoherenceModel
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure


stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

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
            from nltk import pos_tag
            from nltk import word_tokenize
            text = word_tokenize(textblock)
            service2 = pos_tag(text)
            return service2

        def word_cloud(textblock):
            from wordcloud import WordCloud
            wc = WordCloud(background_color='white', width=300, height=300, margin=2).generate(textblock)
            plt.figure(figsize=(8,8), facecolor = 'white')
            plt.imshow(wc)
            plt.axis('off') 
            plt.tight_layout(pad=2)
            plt.savefig('word_cloud.png')


        def topic_modelling(textblock):

            df = (textblock_break(textblock))
            def sent_to_words(sentences):
                for sentence in sentences:
                    yield (gensim.utils.simple_preprocess(str(sentence), deacc=True))
            df_words = list(sent_to_words(df))

            bigram = gensim.models.Phrases(df_words, min_count=5, threshold=100)  # higher threshold fewer phrases.
            bigram_mod = gensim.models.phrases.Phraser(bigram)

            trigram = gensim.models.Phrases(bigram[df_words], threshold=100)
            trigram_mod = gensim.models.phrases.Phraser(trigram)

            def remove_stopwords(texts):
                return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

            def make_bigrams(texts):
                return [bigram_mod[doc] for doc in texts]

            def make_trigrams(texts):
                return [trigram_mod[bigram_mod[doc]] for doc in texts]

            def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
                """https://spacy.io/api/annotation"""
                texts_out = []
                for sent in texts:
                    doc = nlp(" ".join(sent))
                    texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
                return texts_out



            df_words_nostops = remove_stopwords(df_words)
            df_words_bigrams = make_bigrams(df_words_nostops)
            nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
            df_lemmatized = lemmatization(df_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
            id2word = corpora.Dictionary(df_lemmatized)
            texts = df_lemmatized
            corpus = [id2word.doc2bow(text) for text in texts]
            lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                        id2word=id2word,
                                                        num_topics=20,
                                                        random_state=100,
                                                        update_every=1,
                                                        chunksize=100,
                                                        passes=10,
                                                        alpha='auto',
                                                        per_word_topics=True)
            return lda_model.print_topics()

        def textblock_break(textblock):
            sentences = textblock.split(".")
            if len(sentences) == 1:
                return sentences
            else:
                sentences.pop(len(sentences) - 1)
                return sentences

        def aspects(textblock):
            import sys
            import spacy
            nlp = spacy.load("en_core_web_sm")

            sentences = (textblock_break(textblock))

            aspects = []
            for sentence in sentences:
                doc = nlp(sentence)
                descriptive_term = ''
                target = ''
                for token in doc:
                    if token.dep_ == 'nsubj' and token.pos_ == 'NOUN':
                        target = token.text
                    if token.pos_ == 'ADJ':
                        prepend = ''
                        for child in token.children:
                            if child.pos_ != 'ADV':
                                continue
                            prepend += child.text + ' '
                        descriptive_term = prepend + token.text
                aspects.append({'aspect': target,
                                'description': descriptive_term})
            return aspects


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
        if data['user_string']['services']['4']:
            op3 = word_cloud(text)

        if data['user_string']['services']['5']:
            op5 = topic_modelling(text)
            output['Topic modelling'] = op5
        if data['user_string']['services']['6']:
            op6 = aspects(text)
            output['Aspect mining'] = op6




        return render_template('results.html',R1= output['Sentiment analysis'],R2 = output['Part of speech'],
                               R3= output['Text classification'],R4 = plt.imshow(op3),
                               R5= output['Topic modelling'], R6= output['Aspect mining'])
    else:
        return redirect(url_for('home'))


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)

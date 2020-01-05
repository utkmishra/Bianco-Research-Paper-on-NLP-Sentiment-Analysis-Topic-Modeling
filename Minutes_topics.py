#%%
import itertools
import os,sys
import pandas as pd
import numpy as np
from collections import defaultdict,Counter
import re,string
import matplotlib.pyplot as plt
import calendar
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import CategorizedPlaintextCorpusReader,stopwords,wordnet
from nltk.tokenize import sent_tokenize,PunktSentenceTokenizer,word_tokenize  
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize.treebank import TreebankWordDetokenizer
import plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import cufflinks as cf
from wordcloud import WordCloud
from nltk.corpus import opinion_lexicon


pos_list=set(opinion_lexicon.positive())
neg_list=set(opinion_lexicon.negative())  
filename_greg = '/Users/LENOVO USER/Desktop/dictionary1.csv'

import csv

fields = []
rows = []
positive_greg=[]
negative_greg=[]
with open(filename_greg, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    fields = next(csvreader)
    for row in csvreader:
        positive_word = row[0].split(",")[0]
        negative_word = row[1].split(",")[0]
        if(positive_word != ""):
            positive_greg.append(positive_word)
        if(negative_word != ""):
            negative_greg.append(negative_word)
            
print(positive_greg)
print(negative_greg)

positive_consolidated_list = list(pos_list) + positive_greg
negative_consolidated_list = list(neg_list) + negative_greg
print(positive_consolidated_list)
print(negative_consolidated_list)

init_notebook_mode(connected=True)
cf.set_config_file(offline=True, world_readable=True, theme='ggplot')

#%%

corpus_root = "/Users/LENOVO USER/Desktop/FedMin"
data_m = CategorizedPlaintextCorpusReader(corpus_root, r'.*\.txt', cat_pattern=r'(\w+)/*', encoding='latin1')
data_fileids = data_m.fileids()


#%%
def corpus_Stats(crp):
    print('Total number of files: '+str(len(crp.fileids())))
    print('Number of paragraphs: '+str(len(crp.paras())))
    print('Number of sentences: '+str(len(crp.sents())))
    print('Number of words: '+str(len(crp.words())))

#corpus_Stats(data_m)

print('\n'+'First file: '+ data_fileids[0])
print('Last file: '+ data_fileids[-1])

#%%
num_para_py = defaultdict(int)
num_word_py = defaultdict(int)

for y in range(1993,2019):
    files = data_m.fileids(str(y))
    files_size = len(files)
    num_para_py[y] += sum([len(data_m.paras(f))for f in files])/files_size
    num_word_py[y] += sum([len(data_m.words(f))for f in files])/files_size
        
para_words = pd.DataFrame([num_para_py,num_word_py],
                          index = ['Average number of paragraphs','Average number of words']).T

#word around groupbed bar charts
trace0 = go.Bar(x = para_words.index, y=para_words['Average number of paragraphs'], 
                name ='Average number of paragraphs ')
trace1 = go.Bar(x = para_words.index, y=[0],showlegend=False,hoverinfo='none')
trace2 = go.Bar(x = para_words.index, y=[0], yaxis='y2',showlegend=False,hoverinfo='none') 
trace3 = go.Bar(x = para_words.index, y=para_words['Average number of words'],
                yaxis='y2',name ='Average number of words' ) 
data = [trace0,trace1,trace2,trace3]#,trace2

layout = go.Layout(barmode='group',
                   legend=dict(x=0, y=1.1,orientation="h"),
                   yaxis=dict(title='Avg. paragraphs'),
                   yaxis2=dict(title ='Avg. words',
                               overlaying = 'y',
                               side='right')
                   )

#fig = go.Figure(data=data, layout=layout)
#py.offline.iplot(fig)
#fig.show()
#%%
df_temp = pd.DataFrame()
df_year = pd.DataFrame()
for y in range(1993,2019):
    files = data_m.fileids(str(y))
    for f in files:
        word_para = [sum([len(s)for s in p]) for p in data_m.paras(f)]
        df_temp = pd.concat([df_temp,pd.Series(word_para,name=f)],axis=1)
    
    df_mean = df_temp.mean(axis=1)
    df_mean.name = y
    df_year= pd.concat([df_year,df_mean],axis=1)
    df_temp = df_temp.iloc[0:0]

window_size = 10

y1 = df_year.loc[:,[y for y in range(1993,2007)]].mean(axis=1).rolling(window_size).mean()
y2 = df_year.loc[:,[y for y in range(2008,2011)]].mean(axis=1).rolling(window_size).mean()
y3 = df_year.loc[:,[y for y in range(2012,2019)]].mean(axis=1).rolling(window_size).mean()

df_joint = pd.DataFrame([y1,y2,y3],index = ['1993-2007','2008-2011','2012-2018']).T

df_joint.iplot(kind='line', filename='cufflinks/cf-simple-line',
              layout = {'title': 'Average words per paragraph',
                        'xaxis':{'title':'Paragraph index'},
                        'yaxis':{'title':'Word count'}
                       },theme='solar')

#%%
def crop_text(raw_text,start_dict,end_dict):

    sloc=[]
    eloc=[]
    
    for item in start_dict.values():
        slst = re.search(item,raw_text)
        
        if slst != None:
            sloc.extend(slst.span())
    
    for item in end_dict.values():
        elst = re.search(item,raw_text)
        
        if elst != None:
            eloc.extend(elst.span())

    return raw_text[min((sloc), default = 0):max((eloc), default = 0)]

def saveFile(fname,year,text):
    main_directory = '/Users/LENOVO USER/Desktop/FedMin/cropped/'
    os.chdir(main_directory)
    directory = main_directory + str(year) + '/'
    
    if not os.path.exists(directory):#check if directory exists
        os.makedirs(directory)
        
    if not os.path.isfile(fname):#check if file name already exists
        os.chdir(directory)
        file= open(fname, 'w', errors = 'ignore')
        file.write(text)
        file.close()
                
doc_start= {}
doc_start[0] = "Staff Review of the Economic Situation"
doc_start[1] = re.compile('The information (reviewed|received|provided)')
doc_start[2] = "The Committee then turned to a discussion of the economic outlook"
doc_start[3] = "The Committee then turned to a discussion of the economic"
#doc_start[3] = "By unanimous vote"
doc_start[4] = "On the recommendation of the Manager"


doc_end ={}
doc_end[0] = re.compile('(At the conclusion of) (this|the) (discussion|meeting)')
doc_end[1] = re.compile('(?i)The Committee voted to authorize')
doc_end[2] = re.compile('(?i)The vote encompassed approval of')
doc_end[3] = "At the conclusion of the Committee's discussion"

for f in data_fileids:
    year,fname = f.split('/')
    cropped_text = crop_text(data_m.raw(f),doc_start,doc_end)
    saveFile(fname,year,cropped_text)
    

corpus_root_cropped = '/Users/LENOVO USER/Desktop/FedMin/cropped/'
data_c = CategorizedPlaintextCorpusReader(corpus_root_cropped, r'.*\.txt', cat_pattern=r'(\w+)/*', encoding='latin1')

corpus_Stats(data_c)

#%%

nltk.download('averaged_perceptron_tagger')


stopwords = nltk.corpus.stopwords.words('english')
newStopWords = ['1','2','3','4','5','6','7','8','9','.',',',';','(',')',':', 'In', 'in', '?', " '", '$', ":", ";", "'s",  'The', 'The ', 'the', 'A', 'mr.' 'mr', 'committee','bank', 'reserve', 'federal', 'board', 'governor', 'system', 'mr.', 'messrs.', 'also', 'president', 'member', 'committee', 'vice', 'chairman', 'wa', 'director', 'secretary', 'manager', 'ms.', 'ha', 'think', 'like', 
'u', 'lot', ']', '[', 'le', '--', 'doe', 'one', 'could', 'yes', 'may',  u'w', ' ', u"'", '-', 'I', 'T', u'b', 'c', u'e', u'f', u'g', 'h', u'j', 'k', u'l', 'n', u'p', u'r', u'v', u'w', "'", '"',
u'"', u"'", 'say', 'would', 'sure', 'well', 'seem', 'seen', 'let', 'ago', 'want', 'see', 'know', 'cant"' "isn't", "couldn't", 't', "it's", "isn't", "Don't", '!', '"', '%', "'", ',', '/', ':', ';', '<', '=', '>', '@',"`", 'dont' "ca n't", 'ca', "n't", "'re", "quarter", "discussion", "conclusion", "january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december", "year"]
stopwords.extend(newStopWords)


remove_items = stopwords + list(string.punctuation)


print('\n'+'\033[1m'+"Stem tokens"+'\033[0m')
stem = PorterStemmer().stem

def stem_text(text,stemmer,remove_list):#takes in what? sentences in the form of lists
    text_lower = map(str.lower,text)
    return [stemmer(i) for i in text_lower if i not in remove_items and not i.isdigit()]

lemma = WordNetLemmatizer().lemmatize

def get_pos(first_tag):

    if first_tag.startswith('J'):
        return wordnet.ADJ
    elif first_tag.startswith('V'):
        return wordnet.VERB
    elif first_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN #lemmatizer default
    
def lemmatize_text(text,lemmatizer,remove_list):
    pos_tags = nltk.pos_tag(text)
    lem_text = [lemmatizer(t,pos=get_pos(p[1])) for t,p in zip(text,pos_tags)]    
    return [i for i in map(str.lower,lem_text) if i not in remove_items and not i.isdigit()]



#%%
idx = []
p_clean = []
for f in data_c.fileids():
    year,month,day = re.search("(\d{4})(\d{2})(\d{2})",f).groups()
    for i,p in enumerate(data_c.paras(f)):
        idx.append((year,calendar.month_abbr[int(month)],day,i)) #Year,Month,Day,Paragraph number
        flat_p = list(itertools.chain(*p))
        clean_text_s = stem_text(flat_p,stem,remove_items)
        clean_text_l = lemmatize_text(flat_p,lemma,remove_items)
        p_clean.append([clean_text_s,clean_text_l,flat_p])

df_clnpara = pd.DataFrame(data = np.c_[p_clean],
                          index = pd.MultiIndex.from_tuples(idx), 
                          columns = ['clean text s','clean text l','old text'])

#%%
#Most frequent words in corpus
stemmed_values = df_clnpara['clean text l']

wordcloud = WordCloud(collocations=False,
                      background_color='white',
                      mode='RGB',
                      width=1600, height=800).generate((" ").join(itertools.chain(*stemmed_values)))

fig = plt.figure(figsize=(16,8),dpi=200)
plt.axis('off')
plt.imshow(wordcloud, interpolation="bilinear")
plt.show()

#%%
vec = CountVectorizer(tokenizer = lambda x: x,lowercase = False, encoding="ascii", decode_error='ignore', ngram_range=(2, 3), token_pattern='/[A-Za-z0-9_]+(?=\s+)/')
vec_fitted = vec.fit_transform(list(stemmed_values))

num_topics = 7

lda = LatentDirichletAllocation(n_components = num_topics,
                                max_iter = 10,
                                learning_method = 'online',
                                learning_decay =0.9,random_state =10).fit(vec_fitted) 
                               # learning_offset=5,
                                
                                

lda_t = lda.transform(vec_fitted)

df_clnpara['t_score'] = list(lda_t)

def summarize_topics(model, feature_names, no_top_words):
    topics = pd.DataFrame()
    
    for topic_idx, topic in enumerate(model.components_):
        top_words = [feature_names[i]for i in topic.argsort()[:-no_top_words - 1:-1]]
        topics["Topic %d:" % (topic_idx)] = top_words
    
    return topics

df_results = summarize_topics(lda,vec.get_feature_names(),100)

minutes_topics = df_results
minutes_lda = lda_t

#%%

## TO DETERMINE THE BEST MODEL, TAKES LONG TIME,

# Define Search Param
from sklearn.model_selection import GridSearchCV
search_params = {'n_components': [7, 8, 10, 12, 15], 'learning_decay': [.5, .7, .9]}
# Init the Model
lda = LatentDirichletAllocation()
# Init Grid Search Class
model = GridSearchCV(lda, param_grid=search_params)
# Do the Grid Search
model.fit(vec_fitted)
# Best Model
best_lda_model = model.best_estimator_
# Model Parameters
print("Best Model's Params: ", model.best_params_)
# Log Likelihood Score
print("Best Log Likelihood Score: ", model.best_score_)
# Perplexity
print("Model Perplexity: ", best_lda_model.perplexity(vec_fitted))




#%%
trace = go.Table(
    header=dict(values=list(df_results.columns),
                fill = dict(color='#C2D4FF'),
                align = ['left'] * 5),
    cells=dict(values=df_results.values.T,
               fill = dict(color='#F5F8FF'),
               align = ['left'] * 5))

data = dict(data=[trace],layout= {'title':'Top 100 Words per Topic, Minutes, ngram = 2-3, learning_decay =0.9 #of topics =7 ','width':1400,'height':3000})
py.offline.iplot(data)

py.io.write_html(data, file='Minutes_TopicWords.html', auto_open=True)
#%%
#Dictionary with topics


topic_dict = {0 : 'Inflation',
              
              1 : 'Foreign Policy/Trade',
              2 : 'Economic Activity/Outlook',
              3 : 'Labor Condtions/Unemployment', 
              4 : 'Monetary Policy',
              5 : 'Consumption',
              6 : 'Financial Markets'}



topic_dict

#%%
def sum_prop(x):
    x_sum = x.sum(axis=0)
    return (x_sum/x_sum.sum())*100

df_topic_t = df_clnpara.groupby(level=[0,1,2],sort=False)['t_score']\
                        .apply(lambda x: sum_prop(x)).apply(pd.Series)\
                        .rename(columns = topic_dict)

df_topic_m = df_topic_t.groupby(level=[0,1],sort=False).apply(lambda x: sum_prop(x))





df_topic_m.to_csv("/Users/LENOVO USER/Desktop/Minutes_Proportion.csv", sep=",")


#%%
def create_trace_bs(x,y,name):#stacked bar
    trace = dict(x=x,
                 y=y,
                 hoverinfo ='x+y',
                 mode='lines',
                 stackgroup='one',
                 name = name
                 )
    return trace

x = [t[1]+'-'+t[0] for t in df_topic_m.index.values]
tickvals = [x[n] for n in np.where(~df_topic_m.index.get_level_values(0).duplicated(keep='first'))[0]]
ticktext = df_topic_m.index.levels[0]


dataProportion = [create_trace_bs(x,df_topic_m[col].values,col) for col in df_topic_m]

layout = go.Layout(yaxis = dict(ticksuffix='%',showgrid = False),
                   xaxis = dict(type ='category',showgrid = False,
                                tickvals = tickvals,#last of every month
                                ticktext = ticktext,showticklabels=True),
                                   title = 'Topic Proportions For Minutes, Trained on Minutes, ngram = 2-3, learning_decay =0.9, #oftopics =7') 

fig = go.Figure(data=dataProportion,layout=layout)
py.offline.iplot(fig, validate=False)

py.io.write_html(fig, file='Minutes_Topics_Proportions.html', auto_open=True)

def RetrieveScore(tokenized_para,positive,negative): 
    pos_sum = 0
    neg_sum = 0
    score = 0
    if len(tokenized_para) <8:
        return 0
    for word in tokenized_para:
        if word in positive:
                pos_sum +=1
        elif word in negative:
                neg_sum +=1

    try:
        score = ((pos_sum-neg_sum)/(pos_sum+neg_sum)) #should this be 
    except ZeroDivisionError:
        score = 0
    return score

df_clnpara['score'] = df_clnpara.apply(lambda x: RetrieveScore(x['clean text l'],positive_consolidated_list, negative_consolidated_list),axis=1)
df_sentiment = (df_clnpara['score'] * df_clnpara['t_score']).apply(pd.Series).rename(columns = topic_dict)
df_sentiment = pd.concat([df_sentiment.sum(axis=1).rename('Total'),df_sentiment],axis=1)
df_sentiment_m = df_sentiment.groupby(level = [0,1],sort=False).apply(lambda x: x.sum())

def create_trace(df):
    trace_list =[]
    dates = [t[1]+'-'+t[0] for t in df_sentiment_m.index.values]
    for i,n in enumerate(df.columns):
        trace= go.Scatter(x=dates,
                          y=list(df[n]),
                          name=str(n))
        trace_list.append(trace)
    return trace_list


def create_menus(df):
    menus_list =[]
    for i,n in enumerate(df.columns):
        menus_list.append(dict(label=n,method='update',
                               args = [{'visible':list(np.in1d(list(range(0,len(df_sentiment.columns)+1)),i))},
                                       {'title':n}]))
    
    #reset
    menus_list.append(dict(label='Reset',method='update',
                               args = [{'visible':list(np.ones(len(df_sentiment.columns)))},
                                       {'title':'Sentiment'}]))
    
    
    menus = list([dict(type = "buttons",active=-1, buttons = menus_list,
                   direction ='left',
                    showactive = True,
                    x = 0.5,
                    xanchor = 'center',
                    y = 1.11,
                    yanchor = 'top')])
    return menus

layout = go.Layout(dict(title='Sentiment', showlegend=True,
                  updatemenus=create_menus(df_sentiment_m),
                        xaxis = dict(type ='category',
                                    tickvals = tickvals, 
                                    ticktext=ticktext,
                                    showgrid = False,
                                     showline=True),
                        yaxis = dict(showgrid=False,showline=True,title='Sentiment Score')))



fig = dict(data=create_trace(df_sentiment_m), layout=layout)

py.io.write_html(fig, file='Minutes_Topics_Sentiments.html', auto_open=True)

df_sentiment_m.to_csv("/Users/LENOVO USER/Desktop/Minutes_Topics_Sentiment.csv", sep=",")


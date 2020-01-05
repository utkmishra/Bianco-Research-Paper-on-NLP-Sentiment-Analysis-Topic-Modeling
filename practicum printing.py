#%%
import os
import pandas as pd
from nltk.tokenize import word_tokenize
import os.path
from gensim import corpora
from gensim.models import LsiModel
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from gensim.models.coherencemodel import CoherenceModel
import matplotlib.pyplot as plt
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import opinion_lexicon

pos_list=set(opinion_lexicon.positive())
neg_list=set(opinion_lexicon.negative())  

filename_greg = '/Users/LENOVO USER/Desktop/dictionary.csv'
output_alex_csv_minutes = "alex_minutes.csv"
output_greg_csv_minutes = "greg_minutes.csv"
output_alex_csv_transcripts = "alex_transcripts.csv"
output_greg_csv_transcripts = "greg_transcripts.csv"
output_consolidated_csv_minutes = "consolidated_minutes.csv"
output_consolidated_csv_transcripts = "consolidated_transcripts.csv"
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

path_transcripts = "/Users/LENOVO USER/Desktop/Fed_texts"
path_minutes = "/Users/LENOVO USER/Desktop/Fed_Minutes/"

#all_files = os.listdir("/Users/LENOVO USER/Desktop/Fed_texts/")
all_files_minutes = os.listdir(path_minutes)
all_files_transcripts = os.listdir(path_transcripts)
#print(all_files)


def sentiment(word, pos_list, neg_list):
    if word in pos_list:
        return 1
    elif word in neg_list:
        return -1
    return 0

def sentimental_analysis(all_files, path, pos_list, neg_list, output_file):
    os.chdir(path)
    for file in all_files:
        print (file)
        with open(file, encoding="utf8", errors='ignore') as file1:
            tokenized_sents = [word_tokenize(i) for i in file1]
            #print(tokenized_sents)
            # convert list to string
            tokenized_sents_str = ''.join(str(e) for e in tokenized_sents)
    #Extract tokens from strings by using regular expressions 
            from nltk.tokenize import RegexpTokenizer
            tokenizer = RegexpTokenizer(r'\w+')
            tokenizer = tokenizer.tokenize(tokenized_sents_str)
    #print(tokenizer)
            en_stop = set(stopwords.words('english'))
            p_stemmer = PorterStemmer()
            texts = []
            texts = [[word.lower() for word in text.split()] for text in tokenizer]
            stop_words = set(stopwords.words('english'))  
            filtered_sentence = [w for w in tokenized_sents_str if not w in stop_words]  
            filtered_sentence = [] 
    #Stop words were filtered
            for w in tokenizer: 
                if w not in stop_words: 
                    filtered_sentence.append(w)
            lowercase = [item.lower() for item in filtered_sentence]
            nonumbers = [item for item in lowercase if not item.isdigit()]
    #print(nonumbers)
    #Lemmatize text
    #Lemmatization is the process of converting a word to its base form. 
    #The difference between stemming and lemmatization is, lemmatization considers the context and converts the word to its meaningful base form,
    #whereas stemming just removes the last few characters, often leading to incorrect meanings and spelling errors.
            lemmatizer = WordNetLemmatizer()
            lemmatized_words = [lemmatizer.lemmatize(x) for x in nonumbers]
    #print(lemmatized_words)
            remove_items = stopwords.words('english')
            sent_clean = [i for i in map(str.lower,lemmatized_words) if i not in remove_items and not i.isdigit()]
    # Sentimental analysis
            #nltk.download('opinion_lexicon')
            new_list=[] 
            for word in sent_clean:
                new_list.append(sentiment(word,pos_list,neg_list))
            sum=0
            for item in new_list:
                sum+=item
            num_of_positive = 0
            num_of_negative = 0
            num_of_words = 0
            for item in new_list:
                num_of_words+=1
                if item == 1 :
                    num_of_positive+=1
                elif item == -1:
                    num_of_negative+=1
        row = []
        row.append(file)
        row.append(num_of_words)
        row.append(num_of_positive)
        row.append(num_of_negative)
        with open(output_file, 'a' , newline='') as csvFile:
            writer=csv.writer(csvFile)  
            writer.writerow(row)
        csvFile.close()
            
print("Alex List:\n")
print(new_list)

print("Greg List:\n")
print(new_list_greg)

print("Alex's Study:/n/n/n")
sentimental_analysis(all_files_minutes, path_minutes, pos_list, neg_list, output_alex_csv_minutes)
sentimental_analysis(all_files_transcripts, path_transcripts, pos_list, neg_list, output_alex_csv_transcripts)



print("Greg's Study:/n/n/n")
sentimental_analysis(all_files_minutes, path_minutes, positive_greg, negative_greg, output_greg_csv_minutes)
sentimental_analysis(all_files_transcripts, path_transcripts, pos_list, neg_list, output_greg_csv_transcripts)

positive_consolidated_list = list(pos_list) + positive_greg
negative_consolidated_list = list(neg_list) + negative_greg
print("Consolidated Study:/n/n/n")
sentimental_analysis(all_files_minutes, path_minutes, positive_consolidated_list, negative_consolidated_list, output_consolidated_csv_minutes)
sentimental_analysis(all_files_transcripts, path_transcripts, positive_consolidated_list, negative_consolidated_list, output_consolidated_csv_transcripts)







#%%
#print(pos_list)
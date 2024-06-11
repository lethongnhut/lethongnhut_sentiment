import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")

import scipy
from scipy.stats import iqr
from scipy.stats import chi2_contingency
from scipy.stats import chi2
# import statsmodels.api as sm
# from statsmodels.formula.api import ols
# from statsmodels.stats.multicomp import pairwise_tukeyhsd

from underthesea import word_tokenize, pos_tag, sent_tokenize
import regex
# import demoji
# from pyvi import ViPosTagger, ViTokenizer
import string

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression

# for report:
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# vẽ đường cong ROC
from sklearn.metrics import roc_auc_score, roc_curve

import streamlit as st
import pickle
import speech_recognition as sr
from gtts import gTTS
import playsound
import pyaudio
from datetime import date,datetime
import datetime
from time import strftime
import os
import time
import warnings
warnings.filterwarnings("ignore")

#------------------------------------------------------------------------------------------------------------------
def dung():
    text_to_speech("Hẹn gặp lại bạn sau!")
#------------------------------------------------------------------------------------------------------------------
def nhan_text():
    for i in range(3):
        st.write('Mời bạn nói: ')
        text = speech_to_text()
        if text:
            return text.lower()
        elif i < 2:
            speech_to_text("Tôi không nghe rõ. Bạn nói lại được không!")
    time.sleep(2)
    st.write("Hẹn gặp lại bạn sau!")
    # dung()
    return 0
## end voice
#---------------------------------------------------------------------------------------------
def speech_to_text():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        # print("Mời bạn nói: ")
        r.adjust_for_ambient_noise(source)
        audio = r.listen(source,phrase_time_limit=5)
    try:
        text = r.recognize_google(audio,language="vi-VI")
    except:
        text="Xin lỗi! tôi không nhận được giọng nói"
    return str(text)
#---------------------------------------------------------------------------------------------
def text_to_speech(text):
    output = gTTS(text,lang="vi", slow=False)
    date_string = datetime.datetime.now().strftime("%d%m%Y%H%M%S")
    filename = "voice"+date_string+".mp3"
    output.save(filename)
    playsound.playsound(filename)
    os.remove(filename)
#---------------------------------------------------------------------------------------------
@st.cache_data
def df():
    df1 = pd.read_csv('data/Products_Shopee_comments_cleaned_ver01_1.csv')
    df2 = pd.read_csv('data/Products_Shopee_comments_cleaned_ver01_2.csv')
    df = pd.concat([df1, df2], ignore_index=True)
    return df
#---------------------------------------------------------------------------------------------
@st.cache_data
def load_model_Sentiment():
    # Đọc model LogisticRegression()
    pkl_filename = "model/Sentiment_model_best.pkl"
    with open(pkl_filename, 'rb') as file:
        model = pickle.load(file)
    return model
#---------------------------------------------------------------------------------------------
@st.cache_data
def load_model_cv():
    filename = "model/cv_model.pkl"
    with open(filename, 'rb') as file:
        cv = pickle.load(file)
    return cv
#---------------------------------------------------------------------------------------------
## Hàm kiểm tra và tính số lượng, tỷ trọng outliers
def check_outlier(col):
    Q1 = np.percentile(col, 25)
    print('Q1:       ', Q1)
    Q3 = np.percentile(col, 75)
    print('Q3:       ', Q3)
    IQR = scipy.stats.iqr(col)
    print('IQR:      ', IQR)
    highOutliers = (col >= Q3 + 1.5*IQR).sum()
    lowOutliers  = (col <= Q1 - 1.5*IQR).sum()
    print('# Number of upper outliers: ', highOutliers)
    print('# Number of lower outliers: ', lowOutliers)
    print('# Percentage of ouliers:    ', (highOutliers + lowOutliers)/col.shape[0])
#---------------------------------------------------------------------------------------------
## Hàm remove outliers
def remove_outlier(variable, data_param):
# Detection
    Q1 = np.percentile(data_param[variable], 25)
    Q3 = np.percentile(data_param[variable], 75)
    IQR = scipy.stats.iqr(data_param[variable])
    # Upper bound
    upper = np.where(data_param[variable] >= (Q3 + 1.5*IQR))
    # Lower bound
    lower = np.where(data_param[variable] <= (Q1 - 1.5*IQR))
    # Removing the Outliers
    data_param.drop(upper[0], inplace = True)
    data_param.drop(lower[0], inplace = True)
    data_param.reset_index(drop=True, inplace=True)
    return data_param
#---------------------------------------------------------------------------------------------
def process_text(text, emoji_dict, teen_dict, wrong_lst):
    document = text.lower()
    document = document.replace("’", '')
    document = regex.sub(r'\.+', ".", document)
    # Remove punctuation
    document = regex.sub('[^\w\s]', ' ', document)
    punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    for char in punctuation:
        document = document.replace(char, ' ')

    # Remove numbers, only keep letters
    document = regex.sub(r'[\w]*\d+[\w]*', "", document) # document.replace('[\w]*\d+[\w]*', '', regex=True)

    # Some lines start with a space, remove them
    document = regex.sub('^[\s]{1,}', '', document)

    # # Remove multiple spaces with one space
    document = regex.sub('[\s]{2,}', ' ', document)

    # Some lines end with a space, remove them
    document = regex.sub('[\s]{1,}$', '', document)

    # Remove end of line characters
    document = regex.sub(r'[\r\n]+', ' ', document)

    # Remove HTTP links
    document = regex.sub(
        r'((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*', '',
        document)

    new_sentence = ''
    for sentence in sent_tokenize(document):
        # if not(sentence.isascii()):
        ###### CONVERT EMOJICON
        sentence = ''.join(emoji_dict[word] + ' ' if word in emoji_dict else word for word in list(sentence))
        ###### CONVERT TEENCODE
        sentence = ' '.join(teen_dict[word] if word in teen_dict else word for word in sentence.split())
        ###### DEL Punctuation & Numbers
        pattern = r'(?i)\b[a-záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ]+\b'
        sentence = ' '.join(regex.findall(pattern, sentence))
        ###### DEL wrong words
        sentence = ' '.join('' if word in wrong_lst else word for word in sentence.split())
        new_sentence = new_sentence + sentence + '. '
    document = new_sentence
    # print(document)
    ###### DEL excess blank space
    document = regex.sub(r'\s+', ' ', document).strip()
    # ...
    return document
#---------------------------------------------------------------------------------------------
# Chuẩn hóa unicode tiếng việt
def loaddicchar():
    uniChars = "àáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệđìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵÀÁẢÃẠÂẦẤẨẪẬĂẰẮẲẴẶÈÉẺẼẸÊỀẾỂỄỆĐÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴÂĂĐÔƠƯ"
    unsignChars = "aaaaaaaaaaaaaaaaaeeeeeeeeeeediiiiiooooooooooooooooouuuuuuuuuuuyyyyyAAAAAAAAAAAAAAAAAEEEEEEEEEEEDIIIOOOOOOOOOOOOOOOOOOOUUUUUUUUUUUYYYYYAADOOU"

    dic = {}
    char1252 = 'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ'.split(
        '|')
    charutf8 = "à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ".split(
        '|')
    for i in range(len(char1252)):
        dic[char1252[i]] = charutf8[i]
    return dic
#---------------------------------------------------------------------------------------------
# Đưa toàn bộ dữ liệu qua hàm này để chuẩn hóa lại
def covert_unicode(txt):
    dicchar = loaddicchar()
    return regex.sub(
        r'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ',
        lambda x: dicchar[x.group()], txt)
#---------------------------------------------------------------------------------------------
def process_special_word(text):
    new_text = ''
    text_lst = text.split()
    i= 0
    if 'không' in text_lst:
        while i <= len(text_lst) - 1:
            word = text_lst[i]
            #print(word)
            #print(i)
            if  word == 'không':
                next_idx = i+1
                if next_idx <= len(text_lst) -1:
                    word = word +'_'+ text_lst[next_idx]
                i= next_idx + 1
            else:
                i = i+1
            new_text = new_text + word + ' '
    else:
        new_text = text
    return new_text.strip()
#---------------------------------------------------------------------------------------------
def process_postag_thesea(text):
    new_document = ''
    for sentence in sent_tokenize(text):
        sentence = sentence.replace('.','')
        ###### POS tag
        lst_word_type = ['N','Np','A','AB','V','VB','VY','R']
        # lst_word_type = ['A','AB','V','VB','VY','R']
        sentence = ' '.join( word[0] if word[1].upper() in lst_word_type else '' for word in pos_tag(process_special_word(word_tokenize(sentence, format="text"))))
        new_document = new_document + sentence + ' '
    ###### DEL excess blank space
    new_document = regex.sub(r'\s+', ' ', new_document).strip()
    return new_document
#---------------------------------------------------------------------------------------------
def remove_stopword(text, stopwords):
    ###### REMOVE stop words
    document = ' '.join('' if word in stopwords else word for word in text.split())
    #print(document)
    ###### DEL excess blank space
    document = regex.sub(r'\s+', ' ', document).strip()
    return document
#---------------------------------------------------------------------------------------------

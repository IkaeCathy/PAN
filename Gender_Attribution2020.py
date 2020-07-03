import sys
sys.path.append("/anaconda3/lib/python3.7/site-packages")
import numpy as np
import numpy
import nltk
import pickle
import pandas as pd
import glob
import os
import re
import operator
from collections import Counter
from tokenizer2 import *
from PreTokenizer import *
from sstemmer import *
from nltk.stem.porter import PorterStemmer
from nltk.stem import LancasterStemmer, WordNetLemmatizer
from itertools import groupby
from nltk.collocations import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from scipy.cluster.vq import whiten
from nltk.tokenize.toktok import ToktokTokenizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import xml.etree.ElementTree as ET
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import time
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
word_tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
Porter_stemmer = PorterStemmer()
Lancaster_stemmer = LancasterStemmer()
WordNet_lemmatizer = WordNetLemmatizer()
start = time. time()

print(sys.version)


#Train_path="/Users/catherine/Desktop/NLP/PAN_Datasets/PAN2019/TIRA_ikae_2.zip"#"trainfolder"
Train_path = "/Users/catherine/Desktop/NLP/PAN_Datasets/PAN2020/pan20-author-profiling-training-2020-02-23/ENGLISH/en"
Test_path= "/Users/catherine/Desktop/NLP/PAN_Datasets/PAN2020/pan20-author-profiling-training-2020-02-23/ENGLISH/en"

#Train_path="media/training-datasets/author-profiling/pan19-author-profiling-training-dataset-2019-02-18/en"#"trainfolder"
train_truth_path =  "truth.txt"


def author_profiling(inputFolder, outputFolder):

        def read_type_gender(Train_path, truth_path):
            files =   [file for file in glob.iglob(os.path.join(Train_path ,'*.txt'))][0]
            print(files)
            with open(files, mode="r") as f:
                list_of_files=[line for line in f]
                list_of_files1=[file for file in list_of_files]
                #list_of_files1 =([line for line in f])
                fileID =([(line.split(":::", 2))[0] for line in list_of_files1])
                print(fileID[1])
                filetype =([(line.split(":::", 2))[1] for line in list_of_files1])
                print(filetype[10])
            return fileID, filetype

        """
        reading one document at a time........................................................
        """
        def list_xml_file(mypath, truth_path):
                only_train_xml_files =  (read_type_gender(mypath, truth_path))[0]
                xml_file_path =  [(mypath + "/" + file + ".xml") for file in only_train_xml_files]
                return xml_file_path
                
                
        def read_xml_files(xml):#extracts the file contents from an xml file
                 tree = ET.parse(xml)
                 root = tree.getroot()              
                 contents =[content.text for content in root.findall('.//document')]##2019
                 return contents


        ##one training/testing document content with corresponding type..................................................
        def extract_txt(mypath, truth_path):
            df=pd.DataFrame(read_xml_files(mypath, truth_path))
            authortype = [a_type for a_type in df[0]]
            authortext = [text for text in df[1]]
            return authortype, authortext



        ##All training and test text...creates one list of tokens for each document, ..................................................
        def extract_tokens(contents):
                item = ([' '.join(items.split()) for items in contents if items is not None])
                non_bmp_map = dict.fromkeys(range(0x10000, sys.maxunicode + 1), 0xfffd)
                item = [(items.translate(non_bmp_map)) for items in item]
                item = [items.replace("\\u","") for items in item]
                text_per_person = [cleaningLinePAN(items) for items in item]
                text_per_person=[tokenizeLineUTF8(items) for items in text_per_person]
                text_per_person=[[word.lower() for word in items] for items in text_per_person]
                text_per_person =[[WordNet_lemmatizer.lemmatize(word,'v') for word in items] for items in text_per_person]
                text_per_person=[[SStemmer().stem(word) for word in items] for items in text_per_person]
                text_per_person= [item for sublist in text_per_person for item in sublist]
                return text_per_person

        ##create feature vector from the extracted token list.....................................................................................
        def WordFeatures(word_list, all_training_text):
            fvs_words = np.array([(all_training_text.count(word))/(len(all_training_text)) for word in word_list]).astype(np.float64)
            return fvs_words

        """
        reading many documents at once........................................................
        """
        def read_xml_files1(mypath, truth_path):#extracts the file contents of all files at once with coresponding fileID
            author_gender = (read_type_gender(mypath, truth_path))[1]
            only_train_xml_files =  (read_type_gender(mypath, truth_path))[0]
            only_train_xml_files =  [(mypath + "/" + file + ".xml") for file in only_train_xml_files]
            file_contents=[]
            for f in range(len(only_train_xml_files)):
                 tree = ET.parse(only_train_xml_files[f])
                 root = tree.getroot()              
                 contents =[content.text for content in root.findall('.//document')]##2014
                 file_contents.append((author_gender[f], contents))
            return file_contents


        ##All training and test text, match each extracted text with its corresponding gender ..................................................
        def extract_txt1(mypath, truth_path):
            df=pd.DataFrame(read_xml_files1(mypath, truth_path))
            authorgender = [gender for gender in df[0]]
            authortext = [text for text in df[1]]
            return authorgender, authortext



        ##All training and test ,creates a list containing lists of tokens, each list of tokens represents a document..................................................
        def extract_tokens1(mypath, truth_path):
            authorgender = (extract_txt1(mypath, truth_path))[0]
            authortext = (extract_txt1(mypath, truth_path))[1]
            all_tokens_per_person=[]
            for idx in range(len(authorgender)):
                item = ([' '.join(items.split()) for items in authortext[idx] if items is not None])
                non_bmp_map = dict.fromkeys(range(0x10000, sys.maxunicode + 1), 0xfffd)
                item = [(items.translate(non_bmp_map)) for items in item]
                item = [items.replace("\\u","") for items in item]
                text_per_person = [cleaningLinePAN(items) for items in item]
                text_per_person=[tokenizeLineUTF8(items) for items in text_per_person]
                text_per_person=[[word.lower() for word in items] for items in text_per_person]
                text_per_person =[[WordNet_lemmatizer.lemmatize(word,'v') for word in items] for items in text_per_person]
                text_per_person=[[SStemmer().stem(word) for word in items] for items in text_per_person]
                text_per_person= [item for sublist in text_per_person for item in sublist]
                all_tokens_per_person.append(text_per_person)
            return all_tokens_per_person
        """
        write output to xml file.....................................................................................
        """
        def writeDecisionIntoFile(aUserID, decisionBH,  aDir=''):
           #myFile = codecs.open(aDir+aUserID+'.xml', encoding='utf-8', mode="w")
           myFile = open(aDir+aUserID+'.xml', encoding='utf-8', mode="w")
           aString = '<author id=\"'+aUserID+'\"\n   lang=\"en\"\n   type=\"'
           if (decisionBH == 1):
              aString = aString + '0\"\n'
           elif (decisionBH == 2):
              aString = aString + '1\"\n'  
        
           aString = aString +'\n/>\n'
           myFile.write(aString)
           myFile.close()
           return(True)

        """
        create feature vector from the training set.....................................................................................
        """

        def WordFeatures1(word_list, all_training_text):
            fvs_words = np.array([[author.count(word) for word in word_list]
                                   for author in all_training_text]).astype(np.float64)
            # normalise by dividing each row by number of tokens for each author........
            fvs_words /= np.c_[np.array([len(author) for author in all_training_text])]

            return fvs_words


        """
        From the saved csv file, recover the saved features to be used...............................................................
        """
        
        """
        From the saved csv file, recover the saved features to be used...............................................................
        """
        import csv
        word_list=[]

        txt_files =[ "/Users/catherine/Desktop/NLP/PAN_Datasets/PAN2020/pan20-author-profiling-training-2020-02-23/ENGLISH/PAN2020_tweet_0_vocubulary.csv", "/Users/catherine/Desktop/NLP/PAN_Datasets/PAN2020/pan20-author-profiling-training-2020-02-23/ENGLISH/PAN2020_tweet_1_ vocubulary.csv"]

        print("txt_files of features used",txt_files)
        for txt_file in txt_files:
            with open(txt_file, mode="r", encoding="utf-8") as f:
                reader = csv.reader(f, delimiter=",")
                next(reader) # skip header
                word_list1 =  [r[0] for r in reader]
                print(txt_file, len(word_list1))
                word_list = word_list + word_list1
        print("length of features used =", len(word_list))
        word_list=set(word_list)
        print("set_length of features used =", len(word_list))

        

        
        
                
        """
        Prepare the training and test sets to be parsed to the second classifier............................................................
        """
        def extract_type(Train_path, train_truth_path):
                only_train_xml_files =read_type_gender(Train_path, train_truth_path)
                training=[]
                y_train1=[]
                for i in range(len(only_train_xml_files[0])):
                        if (only_train_xml_files[2])[i]!='bot\n':
                                y_train1.append((only_train_xml_files[2])[i])   
                                xml=(Train_path + "/" + (only_train_xml_files[0])[i] + ".xml") 
                                contents=read_xml_files(xml)
                                all_training_text=extract_tokens(contents)
                                feature=WordFeatures(word_listHU, all_training_text)
                                training.append(feature)
                return y_train1, training

        """
        Prepare the training and test sets to be parsed to the classifies............................................................
        """
        all_training_text = extract_tokens1(Train_path, train_truth_path)
        X_train=WordFeatures1(word_list, all_training_text)
        X_train = np.nan_to_num(X_train)
        y_train=np.array(extract_txt1(Train_path, train_truth_path)[0])
        print("The length of Train data =",len( X_train), len(y_train))

       



        #testing and writing output files..................................................................
##        aDir="/Users/catherine/Desktop/NLP/PAN Datasets/PAN2019/xml_result_files/"
        count_gender=0
        count_type=0
        aDir= outputFolder#where will the results be written, specify on the terminal
        Test_path=inputFolder# and a directory where the xml test data is comming from
        #test_truth_path= Test_path + "/" + "truthEN-dev.txt"#the name and where the test truth file 
        only_test_xml_files = [file for file in glob.iglob(os.path.join(Test_path ,'*.xml'))]
        fileID=[(f.split(".")[:-1]) for f in os.listdir(Test_path ) if f.endswith('.xml')]
        testing=[]

        ## the  classifier..................................................................................
        
        from sklearn import model_selection, naive_bayes, svm
        from sklearn.linear_model import SGDClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import LinearSVC
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.neural_network import MLPClassifier
        print("AdaBoostClassifier****************************************************************")

        from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
        clf=AdaBoostClassifier()
        #Train the model using the training sets y_pred=clf.predict(X_test)
        clf.fit(X_train, y_train)
        X_test=X_train
        y_test=y_train
        y_pred=clf.predict(X_test)
        print(y_pred)
        AdaBoostClassifiertfidf=([(i,j) for i, j in zip(y_pred, y_test)])
        print(metrics.accuracy_score(y_test, y_pred))
        
        
        
        for i in range(len(y_pred)):
                #print("i=",i)
                aUserID=fileID[i][0]
                print(aUserID, "y_test = ",y_test[i],  "y_pred = ",y_pred[i], y_pred[i]=="0\n",y_pred[i]=="1\n",y_pred[i]==["1\n"])
                
                #print(" TYPE prediction y_test, y_pred)",y_test, y_pred)
                if y_pred[i]=="1\n":
                        decisionBH=1
                else:
                        decisionBH=0
                #if y_test==y_pred:#good point to check acc for type
                        #count_type=count_type+1
                        

                
                writeDecisionIntoFile(aUserID, decisionBH, aDir)#='')#wite the output file
        
        print("***********************************************************************")
        end = time. time()
        print("time taken",(end - start)/60)
        return()

if __name__ == '__main__':
         print("author profilling")

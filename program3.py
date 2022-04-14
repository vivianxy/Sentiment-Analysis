#! /usr/bin/env python3
# coding=utf-8
import numpy as np
import pandas as pd
from string import punctuation
def accuracy(test_Y, pre, test_X):
    accuracy = np.sum(test_Y==pre) / len(test_X)
    return accuracy
class NaiveBayesClassifier():
    def zero_array(self,x, y):
        self.x=x
        self.y=y
        matrix = [None] * x
        for i in range(len(matrix)):
            matrix[i] = [0]*y
        return matrix
    def bag_of_words(self, data, vocabulary):
        #create an array with 0.
        new_arr = np.array(self.zero_array(len(data),len(vocabulary)))
        for i in range(len(data)):
            for j in data[i]:
                new_arr[i][np.where(j==vocabulary)]=1
        return new_arr
    def Calculation(self, train_X, train_Y, vocabulary):
        self.bow_train = self.bag_of_words(train_X, vocabulary)
        #calculate the P(train_Y)
        self.P_y_zero = len(self.bow_train[train_Y==0]) / (len(self.bow_train[train_Y==0]) + len(self.bow_train[train_Y==1]))
        self.P_y_one = len(self.bow_train[train_Y==1]) / (len(self.bow_train[train_Y==0]) + len(self.bow_train[train_Y==1]))
        #Calculate the P(Xj=uj|y=v)
        self.word_y_zero = (np.sum(self.bow_train[train_Y==0], axis=0) + 1)/(len(self.bow_train[train_Y==0]))
        self.word_y_one = (np.sum(self.bow_train[train_Y==1], axis=0) + 1)/(len(self.bow_train[train_Y==1]))
    #Classification step(page31,32)
    #Y_predict=argmax[log(P(Y=v))+sum(log(P(Xj=uj|Y=v)))]
    def p_predit(self, x, pos_or_neg):
        if pos_or_neg == "neg":
            zero = np.sum(x*np.log(self.word_y_zero)) + np.log(self.P_y_zero)
            return zero
        else:
            one = np.sum(x*np.log(self.word_y_one)) + np.log(self.P_y_one)
            return one
    #Classification step(page33)
    def Classification_step(self, x):
        if self.p_predit(x, "pos") <= self.p_predit(x,"neg"):
            return 0
        else:
            return 1
    def predict(self, X, vocabulary):
        self.bow_test = self.bag_of_words(X, vocabulary)
        preds = [self.Classification_step(x) for x in self.bow_test]
        return np.array(preds)

def main():
    train = np.array(pd.read_csv("trainingSet.txt",sep="\t"))
    test = np.array(pd.read_csv("testSet.txt",sep="\t"))
    train_X, train_Y = np.transpose(train)
    test_X, test_Y = np.transpose(test)
    translator = str.maketrans('','',punctuation)
    t1 = [np.array(str(x).translate(translator).lower().split()) for x in train_X]
    train_X = np.array(t1)
    t2 = [np.array(str(x).translate(translator).lower().split()) for x in test_X]
    test_X = np.array(t2)
    train=(train_X, train_Y)
    test=(test_X, test_Y)
    vocabulary=np.unique(np.array(str(train_X).translate(translator).lower().split()))
    train_X, train_Y = train
    test_X, test_Y = test
    classifier = NaiveBayesClassifier()
    classifier.Calculation(train_X, train_Y, vocabulary)
    predit_train = classifier.predict(train_X, vocabulary)
    predit_test = classifier.predict(test_X, vocabulary)
    train_accuracy = accuracy(train_Y, predit_train, train_X)
    test_accuracy = accuracy(test_Y, predit_test, test_X)
    with open("pre_train.txt","w") as fp:
        print(*vocabulary, sep=",", file=fp)
        for i in range(len(classifier.bow_train)):
            if(train_Y == []):
                print(*classifier.bow_train[i], sep=",", file=fp)
            else:
                print(*classifier.bow_train[i], train_Y[i], sep=",", file=fp) 
    with open("pre_test.txt","w") as fp:
        print(*vocabulary, sep=",", file=fp)
        for i in range(len(classifier.bow_test)):
            if(test_Y == []):
                print(*classifier.bow_test[i], sep=",", file=fp)
            else:
                print(*classifier.bow_test[i], train_Y[i], sep=",", file=fp)
    with open("results.txt","w") as fp:
        print("Result:trainingSet.txt and testSet.txt:",file=fp)
        print("\tTrain Accuracy:", train_accuracy, file=fp)
        print("\tTest Accuracy:", test_accuracy, file=fp)
main()
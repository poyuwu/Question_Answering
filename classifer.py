# coding=utf-8
import json
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import model as classifer
import numpy as np
import sys
#from glob import glob
#from pythonrouge import pythonrouge
#import argparse
#parser = argparse.ArgumentParser()
#parser.add_argument('--mode',type=str,default='train',help='train/test')
#ROUGE = "/home/poyuwu/github/ROUGE/ROUGE-1.5.5.pl" #ROUGE-1.5.5.pl
#data_path = "/home/poyuwu/github/ROUGE/data" #data folder in RELEASE-1.5.5
from nltk import word_tokenize,sent_tokenize
#import nltk
import re
#import string
regex = re.compile('[%s]' % re.escape(u'!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\“”©›®'))#string.prouncation
def remove_urls(vTEXT):
    vTEXT = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', 'tag2url', vTEXT, flags=re.MULTILINE)
    vTEXT = re.sub(ur'—|–|•|…|-|°', ' ', vTEXT, flags=re.MULTILINE)
    vTEXT = re.sub(ur'’|‘',"'",vTEXT)
    vTEXT = re.sub(ur'ˀ',"",vTEXT)
    vTEXT = re.sub('From Wikipedia, the free encyclopedia\.?','',vTEXT,flags=re.IGNORECASE)
    return(vTEXT)
def pad_array(arr,seq_len=None): #2D
    if seq_len is None:
        M = max(len(a) for a in arr)
        return np.array([a + [0] * (M - len(a)) for a in arr])
    else:
        output = np.zeros((len(arr),seq_len))
        for i in range(len(arr)):
            for j in range(seq_len):
                if j < len(arr[i]):
                    output[i][j] = arr[i][j]
        return output.astype(int)
id_mapping = {"PAD_ID": 0,"EOS_ID":1,"UNK_ID":2,"GO":3}#,"tag2number":6}
id2word = ["PAD_ID","EOS_ID","UNK_ID","GO"]#,"tag2number"]
count = len(id2word)
#read data
train_facts, train_question, train_answer = [], [], []
dev_facts, dev_question, dev_answer = [], [], []
d_max_len , q_max_len , d_max_sent = 55, 24, 48
#d_max_len,q_max_len= 1215,24
a_max_len = 28
train_d_length,train_q_length ,train_a_length,train_d_sent_len = [],[],[],[]
train_q, train_d, train_d_len, train_q_len,train_d_sent, train_class = [],[],[],[],[],[]
dev_d_length,dev_q_length,dev_a_length,dev_d_sent_len = [],[],[],[]
dev_q, dev_d, dev_d_len, dev_q_len, dev_d_sent, dev_class = [],[],[],[],[],[]
with open('train_v1.1.json') as f:
    for line in f:
        line = json.loads(line)
        if len(line['answers']) == 0:#No answer
            continue
        #answer
        ans_temp = [3]
        for token in word_tokenize(remove_urls(line['answers'][0].lower())):
            token = regex.sub(u'',token)
            if token == u'':
                continue
            #if token.isdigit():
            #    ans_temp.append(6)
            if token in id_mapping:
                ans_temp.append(id_mapping.get(token))
            else:
                ans_temp.append(count)
                id_mapping.update({token:count})
                id2word.append(token)
                count += 1
        ans_temp.append(1)
        if len(ans_temp) > a_max_len:
            continue
        #question
        q_temp = []
        for token in word_tokenize(remove_urls(line['query'].lower())):
            token = regex.sub(u'',token)
            if token == u'':
                continue
            #if token.isdigit():
            #    q_temp.append(6)
            if token in id_mapping:
                q_temp.append(id_mapping.get(token))
            else:
                q_temp.append(count)
                id_mapping.update({token:count})
                id2word.append(token)
                count += 1
        if len(q_temp) > q_max_len:
            continue
        #document
        for passages in line['passages']:
            temp_facts = []
            for sent in sent_tokenize(remove_urls(passages['passage_text'])):
                temp = []
                for token in word_tokenize(sent):
                    token = regex.sub(u'',token)
                    if token == u'':
                        continue
                    token = token.lower()
                    #if token.isdigit():
                    #    temp.append(6)
                    if token in id_mapping:
                        temp.append(id_mapping.get(token))
                    else:
                        temp.append(count)
                        id_mapping.update({token:count})
                        id2word.append(token)
                        count += 1
                #temp.append(1)
                #if len(temp) > 100:
                    #print sent#map(lambda x: id2word[x],temp)
                #train_q, train_d, train_d_len, train_q_len, train_class = [],[],[],[]]
                temp_facts.append(temp[:55])
            if len(temp_facts) > 48:
                continue
            train_d.append(list(pad_array(temp_facts,d_max_len))+[[0]*d_max_len]*(d_max_sent-len(temp_facts)))
            train_d_sent.append(map(len,temp_facts)+[0]*(d_max_sent-len(temp_facts)))
            train_d_len.append(len(temp_facts))
            train_q.append(q_temp)
            train_q_len.append(len(q_temp))
            #d_max_sent = max(d_max_sent,len(temp_facts))
            if passages['is_selected']  == 1:
                train_class.append([0,1])
            else:
                train_class.append([1,0])
        #if max(map(len,temp_facts)) > 55:
        #    continue
        #temp.append(1)
        #document
        #d_max_len = max(map(len,temp_facts)+[d_max_len])
        train_facts.append(list(pad_array(temp_facts,d_max_len))+[[0]*d_max_len]*(d_max_sent-len(temp_facts)))
        train_d_length.append(len(temp_facts))
        train_d_sent_len.append(map(len,temp_facts)+[0]*(d_max_sent-len(temp_facts)))
        #question
        train_question.append(q_temp)
        train_q_length.append(len(q_temp))
        #q_max_len = max(q_max_len,len(q_temp))
        #ans
        train_a_length.append(len(ans_temp))
        train_answer.append(ans_temp)
train_d = np.array(train_d)
train_q = pad_array(train_q,q_max_len)
train_q_len = np.array(train_q_len)
train_d_sent = np.array(train_d_sent)
train_d_len = np.array(train_d_len)
train_class = np.array(train_class)
with open('dev_v1.1.json') as f:
    for line in f:
        line = json.loads(line)
        if len(line['answers']) == 0:#No answer
            continue
        #answer
        ans_temp = [3]
        for token in word_tokenize(remove_urls(line['answers'][0].lower())):
            token = regex.sub(u'',token)
            if token == u'':
                continue
            #if token.isdigit():
            #    ans_temp.append(6)
            if token in id_mapping:
                ans_temp.append(id_mapping.get(token))
            else:
                ans_temp.append(count)
                id_mapping.update({token:count})
                id2word.append(token)
                count += 1
        ans_temp.append(1) #EOS
        if len(ans_temp) > a_max_len:
            continue
        #question
        temp_q = []
        for token in word_tokenize(remove_urls(line['query'].lower())):
            token = regex.sub(u'',token)
            if token == u'':
                continue
            #if token.isdigit():
            #    temp_q.append(6)
            if token in id_mapping:
                temp_q.append(id_mapping.get(token))
            else:
                temp_q.append(count)
                id_mapping.update({token:count})
                id2word.append(token)
                count += 1
        temp_q = temp_q[:q_max_len]
        #if len(temp_q) > q_max_len:
        #    continue
        #document
        for passages in line['passages']:
            temp_facts = []
            for sent in sent_tokenize(remove_urls(passages['passage_text'])):
                temp = []
                for token in word_tokenize(sent):
                    token = regex.sub(u'',token)
                    if token == u'':
                        continue
                    token = token.lower()
                    #if token.isdigit():
                    #    temp.append(6)
                    if token in id_mapping:
                        temp.append(id_mapping.get(token))
                    else:
                        temp.append(count)
                        id_mapping.update({token:count})
                        id2word.append(token)
                        count += 1
                #if len(temp) > 100:
                #    print map(lambda x: id2word[x],temp)
                if len(temp) > 55:
                    temp = temp[:55]
                if len(temp_facts) > 48:
                    break
                temp_facts.append(temp)
            dev_d.append(list(pad_array(temp_facts,d_max_len))+[[0]*d_max_len]*(d_max_sent-len(temp_facts)))
            dev_d_sent.append(map(len,temp_facts)+[0]*(d_max_sent-len(temp_facts)))
            dev_d_len.append(len(temp_facts))
            dev_q.append(q_temp)
            dev_q_len.append(len(q_temp))
            if passages['is_selected']  == 1:
                dev_class.append([0,1])
            else:
                dev_class.append([1,0])
        #document
        #d_max_len = max(map(len,temp_facts)+[d_max_len])
        dev_facts.append(list(pad_array(temp_facts,d_max_len))+[[0]*d_max_len]*(d_max_sent-len(temp_facts)))
        dev_d_length.append(len(temp_facts))
        d_max_sent = max(d_max_sent,len(temp_facts))
        dev_d_sent_len.append(map(len,temp_facts)+[0]*(d_max_sent-len(temp_facts)))
        #d_max_len = max(d_max_len,len(temp))
        #dev_facts.append(temp_facts)
        #dev_d_length.append(len(temp_facts))
        #d_max_sent = max(d_max_sent,len(temp_facts))
        #question
        dev_question.append(temp_q)
        dev_q_length.append(len(temp_q))
        q_max_len = max(q_max_len,len(temp_q))
        #ans
        dev_a_length.append(len(ans_temp))
        dev_answer.append(ans_temp)
dev_d = np.array(dev_d)
dev_q = pad_array(dev_q,q_max_len)
dev_q_len = np.array(dev_q_len)
dev_d_sent = np.array(dev_d_sent)
dev_d_len = np.array(dev_d_len)
dev_class = np.array(dev_class)

del train_d_length,train_q_length ,train_a_length,train_d_sent_len
del dev_d_length,dev_q_length,dev_a_length,dev_d_sent_len
del train_facts, train_question, train_answer
del dev_facts, dev_question, dev_answer
'''
train_facts = np.array(train_facts)
train_question = pad_array(train_question,q_max_len)
train_answer = pad_array(train_answer,a_max_len)#a_max_len = 40
train_d_length = np.array(train_d_length)
train_q_length = np.array(train_q_length)
train_a_length = np.array(train_a_length)
train_d_sent_len = np.array(train_d_sent_len)
dev_facts = np.array(dev_facts)
dev_question = pad_array(dev_question,q_max_len)
dev_answer = pad_array(dev_answer,a_max_len)#a_max_len = 40
dev_d_length = np.array(dev_d_length)
dev_q_length = np.array(dev_q_length)
dev_a_length = np.array(dev_a_length)
dev_d_sent_len = np.array(dev_d_sent_len)
'''
num_symbol = len(id2word)
rnn_size = 512
layer = 1
embedding_size = 500
batch_size = 64
index_list = np.array(range(len(train_d)))
nb_batch = len(index_list)/batch_size

model = classifer.Model(d_max_length=d_max_len,q_max_length=q_max_len,a_max_length=a_max_len,num_symbol=num_symbol,rnn_size=rnn_size,layer=layer,embedding_size=embedding_size,d_max_sent=d_max_sent)
model.build_model()
print d_max_len,q_max_len,a_max_len,num_symbol,rnn_size,layer,embedding_size,d_max_sent
print 'training QA generator'
#nb_batch = len(index_list)/batch_size
for epoch in range(150):
    avg = 0.
    np.random.shuffle(index_list)
    ### train on msmarco
    tp, tn, fp, fn = 0., 0., 0., 0.
    for num in range(nb_batch):
        opti = model.update
        loss = model.output_net['loss']
        _, cost,tmp1, tmp2,tmp3,tmp4 = model.sess.run([opti,loss,model.tp,model.tn,model.fp,model.fn],
                feed_dict={
                    model.input_net['d']:train_d[index_list[num*batch_size:num*batch_size+batch_size]],
                    model.input_net['d_mask']:train_d_sent[index_list[num*batch_size:num*batch_size+batch_size]],
                    model.input_net['d_sent_mask']:train_d_len[index_list[num*batch_size:num*batch_size+batch_size]],
                    model.input_net['q']:train_q[index_list[num*batch_size:num*batch_size+batch_size]],
                    model.input_net['q_mask']:train_q_len[index_list[num*batch_size:num*batch_size+batch_size]],
                    model.input_net['labels']:train_class[index_list[num*batch_size:num*batch_size+batch_size]],
                    model.input_net['drop']:0.6})
        avg += cost
        tp += tmp1
        tn += tmp2
        fp += tmp3
        fn += tmp4
        sys.stdout.write(str(epoch)+"\t traininig loss "+str(avg/(num+1))+"\r")
        sys.stdout.flush()
    sys.stdout.write(str(epoch)+"\t traininig loss "+str(avg/nb_batch)+"\t"+"\n")
    accuracy = (tp + tn) / (tp + fp + fn + tn)
    precision = tp / (tp + fp+1e-10)
    recall = tp / (tp + fn)
    fmeasure = (2 * precision * recall) / (precision + recall+1e-10)
    print 'acc: ', accuracy
    print 'precision: ', precision
    print 'recall: ', recall
    print 'fmeasure', fmeasure
    ###test on msmarco
    loss, tp, tn, fp, fn = 0., 0., 0., 0., 0.
    for i in range(len(dev_d)/batch_size +1):
        start = i*batch_size
        end = i*batch_size + batch_size
        if i == len(dev_d)/batch_size:
            end = len(dev_d)
        cost, tmp1, tmp2, tmp3, tmp4 = model.sess.run([model.output_net['loss'],model.tp,model.tn,model.fp,model.fn],
                    feed_dict={
                        model.input_net['d']:dev_d[start:end],
                        model.input_net['d_mask']:dev_d_sent[start:end],#dev_d_length[start:end],
                        model.input_net['d_sent_mask']:dev_d_len[start:end],
                        model.input_net['q']:dev_q[start:end],
                        model.input_net['q_mask']:dev_q_len[start:end],
                        model.input_net['labels']:dev_class[start:end],
                    model.input_net['drop']:1})
        loss += cost * batch_size
        tp += tmp1
        tn += tmp2
        fp += tmp3
        fn += tmp4
        sys.stdout.write(str(epoch)+"\t ms loss:"+str(np.mean(cost))+"\r")
        sys.stdout.flush()
    sys.stdout.write(str(epoch)+"\t ms loss:"+str(loss/len(dev_d))+"\n")
    accuracy = (tp + tn) / (tp + fp + fn + tn)
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn)
    fmeasure = (2 * precision * recall) / (precision + recall+1e-10)
    print 'acc: ', accuracy
    print 'precision: ', precision
    print 'recall: ', recall
    print 'fmeasure', fmeasure

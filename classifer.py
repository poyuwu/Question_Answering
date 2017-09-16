# coding=utf-8
import json
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import model as classifer
import numpy as np
import sys
import gc
gc.enable()
#from glob import glob
#from pythonrouge import pythonrouge
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--rnn_size',type=int,default=512,help='cell size')
parser.add_argument('--embedding_size',type=int,default=500,help='embedding')
parser.add_argument('--replace',type=int,default=2,help='replace digit')
parser.add_argument('--device',type=str,default="1",help='GPU device')
parser.add_argument('--batch_size',type=int,default=16,help='batch size')
parser.add_argument('--encode_type',type=str,default="cnn",help='encode document type')
args = parser.parse_args()
print args
os.environ["CUDA_VISIBLE_DEVICES"] = args.device
#ROUGE = "/home/poyuwu/github/ROUGE/ROUGE-1.5.5.pl" #ROUGE-1.5.5.pl
#data_path = "/home/poyuwu/github/ROUGE/data" #data folder in RELEASE-1.5.5
from nltk import word_tokenize, sent_tokenize
def ListtoString(l1):
    res = ""
    for i in l1:
        if i == id_mapping["EOS_ID"]:
            break
        elif i == id_mapping["PAD_ID"] or i == id_mapping["GO"]:
            continue
        res += id2word[i] + " "
    return res[:-1]
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
    vTEXT = re.sub('‍|‎|​|‏','',vTEXT)
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
if args.replace == 1:
    tag2number = len(id2word)
    id2word.append("tag2number")
    id_mapping.update({"tag2number": tag2number})
mapping = {"description": 0,"numeric": 1,"entity": 2,"location": 3,"person": 4}
query_type = ["description","numeric","entity","location","person"]
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
train_group, dev_group = [], []
with open('train_v1.1.json') as f:
    for line in f:
        number_dict = {}
        line = json.loads(line)
        '''
        if len(line['answers']) == 0:#No answer
            continue
        #answer
        ans_temp = [3]
        for token in word_tokenize(remove_urls(line['answers'][0].lower())):
            token = regex.sub(u'',token)
            if token == u'':
                continue
            if token.isdigit():
                if args.replace == 1:
                    ans_temp.append(tag2number)
                    continue
                elif args.replace == 2:
                    if token in number_dict:
                        token = number_dict.get(token)
                    else:
                        number_dict.update({token: "tag2number" + str(len(number_dict))})
                        token = "tag2number" + str(len(number_dict) - 1)
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
        '''
        #question
        q_temp = []
        for token in word_tokenize(remove_urls(line['query'].lower())):
            token = regex.sub(u'',token)
            if token == u'':
                continue
            if token.isdigit():
                if args.replace == 1:
                    q_temp.append(tag2number)
                    continue
                elif args.replace == 2:
                    if token in number_dict:
                        token = number_dict.get(token)
                    else:
                        number_dict.update({token: "tag2number" + str(len(number_dict))  })
                        token = "tag2number" + str(len(number_dict) - 1)
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
        head = len(train_d)
        for passages in line['passages']:
            temp_facts = []
            for sent in sent_tokenize(remove_urls(passages['passage_text'])):
                temp = []
                for token in word_tokenize(sent):
                    token = regex.sub(u'',token)
                    if token == u'':
                        continue
                    token = token.lower()
                    if token.isdigit():
                        if args.replace == 1:
                            temp.append(tag2number)
                            continue
                        elif args.replace == 2:
                            if token in number_dict:
                                token = number_dict.get(token)
                            else:
                                number_dict.update({token: "tag2number" + str(len(number_dict))})
                                token = "tag2number" + str(len(number_dict) - 1)
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
            if len(temp_facts) > 48: # if # of passages is more than 48:
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
        end = len(train_d)
        train_group.append([head,end])
        #if max(map(len,temp_facts)) > 55:
        #    continue
        #temp.append(1)
        #document
        #d_max_len = max(map(len,temp_facts)+[d_max_len])
#        train_facts.append(list(pad_array(temp_facts,d_max_len))+[[0]*d_max_len]*(d_max_sent-len(temp_facts)))
#        train_d_length.append(len(temp_facts))
#        train_d_sent_len.append(map(len,temp_facts)+[0]*(d_max_sent-len(temp_facts)))
        #question
#        train_question.append(q_temp)
#        train_q_length.append(len(q_temp))
        #q_max_len = max(q_max_len,len(q_temp))
        #ans
#        train_a_length.append(len(ans_temp))
#        train_answer.append(ans_temp)
train_d = np.array(train_d)
train_q = pad_array(train_q,q_max_len)
train_q_len = np.array(train_q_len)
train_d_sent = np.array(train_d_sent)
train_d_len = np.array(train_d_len)
train_class = np.array(train_class)

dev_id, dev_type = [], []
with open('dev_v1.1.json') as f:
    for line in f:
        line = json.loads(line)
        number_dict = {}
        if len(line['answers']) == 0:#No answer
            continue
        #answer
        ans_temp = [3]
        for token in word_tokenize(remove_urls(line['answers'][0].lower())):
            token = regex.sub(u'',token)
            if token == u'':
                continue
            if token.isdigit():
                if args.replace == 1:
                    ans_temp.append(tag2number)
                    continue
                elif args.replace == 2:
                    if token in number_dict:
                        token = number_dict.get(token)
                    else:
                        number_dict.update({token: "tag2number"+ str(len(number_dict))})
                        token = "tag2number" + str(len(number_dict) -1)
            if token in id_mapping:
                ans_temp.append(id_mapping.get(token))
            else:
                ans_temp.append(count)
                id_mapping.update({token:count})
                id2word.append(token)
                count += 1
        ans_temp.append(1) #EOS
        if len(ans_temp) > a_max_len:
            ans_temp = ans_temp[:a_max_len - 1]
            ans_temp.append(1)
            #continue
        #question
        q_temp = []
        for token in word_tokenize(remove_urls(line['query'].lower())):
            token = regex.sub(u'',token)
            if token == u'':
                continue
            if token.isdigit():
                if args.replace == 1:
                    q_temp.append(tag2number)
                    continue
                elif args.replace == 2:
                    if token in number_dict:
                        token = number_dict.get(token)
                    else:
                        number_dict.update({token: "tag2number"+ str(len(number_dict))})
                        token = "tag2number" + str(len(number_dict) -1)
            if token in id_mapping:
                q_temp.append(id_mapping.get(token))
            else:
                q_temp.append(count)
                id_mapping.update({token:count})
                id2word.append(token)
                count += 1
        q_temp = q_temp[:q_max_len]
        #if len(q_temp) > q_max_len:
        #    continue
        #document
        head = len(dev_d)
        for passages in line['passages']:
            temp_facts = []
            for sent in sent_tokenize(remove_urls(passages['passage_text'])):
                temp = []
                for token in word_tokenize(sent):
                    token = regex.sub(u'',token)
                    if token == u'':
                        continue
                    token = token.lower()
                    if token.isdigit():
                        if args.replace == 1:
                            temp.append(tag2number)
                            continue
                        elif args.replace == 2:
                            if token in number_dict:
                                token = number_dict.get(token)
                            else:
                                number_dict.update({token: "tag2number"+ str(len(number_dict))})
                                token = "tag2number" + str(len(number_dict) -1)
                    if token in id_mapping:
                        temp.append(id_mapping.get(token))
                    else:
                        temp.append(count)
                        id_mapping.update({token:count})
                        id2word.append(token)
                        count += 1
                #if len(temp) > 100:
                #    print map(lambda x: id2word[x],temp)
                #if len(temp) > 55:
                #    temp = temp[:55]
                temp_facts.append(temp[:55])
                if len(temp_facts) >= 48:
                    break
            dev_d.append(list(pad_array(temp_facts,d_max_len))+[[0]*d_max_len]*(d_max_sent-len(temp_facts)))
            dev_d_sent.append(map(len,temp_facts)+[0]*(d_max_sent-len(temp_facts)))
            dev_d_len.append(len(temp_facts))
            dev_q.append(q_temp)
            dev_q_len.append(len(q_temp))
            if passages['is_selected']  == 1:
                dev_class.append([0,1])
            else:
                dev_class.append([1,0])
        dev_id.append(line['query_id'])
        dev_type.append(line['query_type'])
        end = len(dev_d)
        dev_group.append([head,end-1])
        number_dict.clear()
        del number_dict
        #document
        #d_max_len = max(map(len,temp_facts)+[d_max_len])
        #dev_facts.append(list(pad_array(temp_facts,d_max_len))+[[0]*d_max_len]*(d_max_sent-len(temp_facts)))
        #dev_d_length.append(len(temp_facts))
        #d_max_sent = max(d_max_sent,len(temp_facts))
#        dev_d_sent_len.append(map(len,temp_facts)+[0]*(d_max_sent-len(temp_facts)))
        #d_max_len = max(d_max_len,len(temp))
        #dev_facts.append(temp_facts)
        #dev_d_length.append(len(temp_facts))
        #d_max_sent = max(d_max_sent,len(temp_facts))
        #question
#        dev_question.append(q_temp)
#        dev_q_length.append(len(q_temp))
#        q_max_len = max(q_max_len,len(q_temp))
        #ans
#        dev_a_length.append(len(ans_temp))
        dev_answer.append(ans_temp)
dev_d = np.array(dev_d)
dev_q = pad_array(dev_q,q_max_len)
dev_q_len = np.array(dev_q_len)
dev_d_sent = np.array(dev_d_sent)
dev_d_len = np.array(dev_d_len)
dev_class = np.array(dev_class)

del train_d_length, train_q_length ,train_a_length,train_d_sent_len
del dev_d_length,dev_q_length,dev_a_length,dev_d_sent_len
del train_facts, train_question, train_answer
del dev_facts, dev_question
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
rnn_size = args.rnn_size
layer = 1
embedding_size = args.embedding_size
batch_size = args.batch_size
index_list = np.array(range(len(train_d)))
nb_batch = len(index_list)/batch_size

model = classifer.Model(d_max_length=d_max_len,q_max_length=q_max_len,a_max_length=a_max_len,num_symbol=num_symbol,rnn_size=rnn_size,layer=layer,embedding_size=embedding_size,d_max_sent=d_max_sent,encode_type=args.encode_type)
model.build_model()
print d_max_len,q_max_len,a_max_len,num_symbol,rnn_size,layer,embedding_size,d_max_sent
print 'training QA discriminator'
#nb_batch = len(index_list)/batch_size
for epoch in range(150):
    avg = 0.
    np.random.shuffle(index_list)
    batch_size = args.batch_size
    ### train on msmarco
    tp, tn, fp, fn = 0., 0., 0., 0.
    for num in range(nb_batch):
        opti = model.update
        loss = model.output_net['loss']
        _, cost, tmp1, tmp2, tmp3, tmp4 = model.sess.run([opti,loss,model.tp,model.tn,model.fp,model.fn],
                feed_dict={
                    model.input_net['d']:train_d[index_list[num*batch_size:num*batch_size+batch_size]],
                    model.input_net['d_mask']:train_d_sent[index_list[num*batch_size:num*batch_size+batch_size]],
                    model.input_net['d_sent_mask']:train_d_len[index_list[num*batch_size:num*batch_size+batch_size]],
                    model.input_net['q']:train_q[index_list[num*batch_size:num*batch_size+batch_size]],
                    model.input_net['q_mask']:train_q_len[index_list[num*batch_size:num*batch_size+batch_size]],
                    model.input_net['labels']:train_class[index_list[num*batch_size:num*batch_size+batch_size]],
                    model.input_net['drop']:0.5})
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
    batch_size = 60
    f_out = open('/home_local/poyuwu/QA/result/dev_'+str(epoch) + '.json','w')
    for i in range(len(dev_group)/batch_size):
        start = i*batch_size#dev_group[i*batch_size]
        if i == len(dev_group)/batch_size -1:
            end = len(dev_group) - 1 
        else:
            end = i*batch_size+batch_size - 1
        #start = i*batch_size
        #end = i*batch_size + batch_size
        #if i == len(dev_d)/batch_size:
        #    end = len(dev_d)
        prob, tmp1, tmp2, tmp3, tmp4 = model.sess.run([model.prob[:,1],model.tp,model.tn,model.fp,model.fn],
                    feed_dict={
                        model.input_net['d']:dev_d[dev_group[start][0]:dev_group[end][1]+1],
                        model.input_net['d_mask']:dev_d_sent[dev_group[start][0]:dev_group[end][1]+1],#dev_d_length[start:end],
                        model.input_net['d_sent_mask']:dev_d_len[dev_group[start][0]:dev_group[end][1]+1],
                        model.input_net['q']:dev_q[dev_group[start][0]:dev_group[end][1]+1],
                        model.input_net['q_mask']:dev_q_len[dev_group[start][0]:dev_group[end][1]+1],
                        model.input_net['labels']:dev_class[dev_group[start][0]:dev_group[end][1]+1],
                    model.input_net['drop']:1})
        for j in range(end-start):
            out_dict = {}
            out_dict.update({"query_id": dev_id[start+j]})
            out_dict.update({"query_type": dev_type[start+j]})
            out_dict.update({"query": ListtoString(dev_q[dev_group[start+j][0]])})
            out_dict.update({"answers": [ListtoString(dev_answer[start+j]) ]})
            max_index = np.argmax(prob[dev_group[start+j][0]-dev_group[start][0]:dev_group[start+j][1]-dev_group[start][0]])
            #passage_text = [ListtoString(passages) for passages in dev_d[start+max_index] if not passages[0] == 0]
            out_dict.update({"passages": [{"is_selected": 1 , "passages_text" : ListtoString(passages)} for passages in dev_d[dev_group[start+j][0]+max_index] if not passages[0] == 0] })
            json.dump(out_dict, f_out)
            f_out.write("\n")
            out_dict.clear()
            del out_dict
        #loss += cost * batch_size
        tp += tmp1
        tn += tmp2
        fp += tmp3
        fn += tmp4
        #sys.stdout.write(str(epoch)+"\t ms loss:"+str(np.mean(cost))+"\r")
        sys.stdout.flush()
    #sys.stdout.write(str(epoch)+"\t ms loss:"+str(loss/len(dev_d))+"\n")
    accuracy = (tp + tn) / (tp + fp + fn + tn)
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn)
    fmeasure = (2 * precision * recall) / (precision + recall+1e-10)
    print 'acc: ', accuracy
    print 'precision: ', precision
    print 'recall: ', recall
    print 'fmeasure', fmeasure
    f_out.close()

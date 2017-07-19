# coding=utf-8
import json
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import finaldmn as qaseq
import numpy as np
import sys
#from glob import glob
#from pythonrouge import pythonrouge
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--rnn_size',type=int,default=256,help='cell size')
parser.add_argument('--embedding_size',type=int,default=500,help='embedding')
parser.add_argument('--replace',type=int,default=1,help='replace digit')
parser.add_argument('--device',type=str,default="1",help='GPU device')
parser.add_argument('--batch_size',type=int,default=16,help='batch size')
args = parser.parse_args()
print args
os.environ["CUDA_VISIBLE_DEVICES"] = args.device
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
id_mapping = {"PAD_ID": 0,"EOS_ID":1,"UNK_ID":2,"GO":3,"1":4,"0":5}
id2word = ["PAD_ID","EOS_ID","UNK_ID","GO","1","0"]
# replace all integer(1,2,30,1000,...) to tag2number
if args.replace == 1:
    tag2number = len(id2word)
    id2word.append("tag2number")
    id_mapping.update({"tag2number": tag2number})
mapping = {"description":0,"numeric":1,"entity":2,"location":3,"person":4}
count = len(id2word)
#read data
train_facts, train_question, train_answer = [], [], []
dev_facts, dev_question, dev_answer = [], [], []
d_max_len , q_max_len , d_max_sent = 55, 24, 48 # sentence level
#d_max_len,q_max_len= 1215,24 # word level
a_max_len = 28
train_d_length,train_q_length ,train_a_length,train_d_sent_len = [],[],[],[]
train_id ,train_type = [], []
with open('train_v1.1.json') as f:
    for line in f:
        number_dict = {}
        line = json.loads(line)
        if len(line['answers']) == 0:#No answer
            continue
        #answer
        ans_temp = [3]
        for token in word_tokenize(remove_urls(line['answers'][0].lower())):
            token = regex.sub(u'',token)
            if token == u'':
                continue
            # processing with digit
            if token.isdigit():
                if args.replace == 1:
                    ans_temp.append(tag2number)
                    continue
                elif args.replace == 2:
                    if token in number_dict:
                        token = number_dict.get(token)
                    else:
                        number_dict.update({token:"tag2number_"+str(len(number_dict))})
                        token = "tag2number_" + str(len(number_dict)-1)
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
        #document
        temp_facts = []
        is_selected = False
        for passages in line['passages']:
            if passages['is_selected']  == 1:
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
                                    number_dict.update({token:"tag2number_"+str(len(number_dict))})
                                    token = "tag2number_" + str(len(number_dict)-1)
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
                    temp_facts.append(temp)
                is_selected = True
        if is_selected == False: #or len(temp_facts) > d_max_len: #temp_facts No info
            continue
#        if len(ans_temp) > a_max_len:
#            continue
        #train_question
        temp = []
        for token in word_tokenize(remove_urls(line['query'].lower())):
            token = regex.sub(u'',token)
            if token == u'':
                continue
            if token.isdigit():
                if args.replace == 1:
                    temp.append(tag2number)
                    continue
                elif args.replace == 2:
                    if token in number_dict:
                        token = number_dict.get(token)
                    else:
                        number_dict.update({token:"tag2number_"+str(len(number_dict))})
                        token = "tag2number_" + str(len(number_dict)-1)
            if token in id_mapping:
                temp.append(id_mapping.get(token))
            else:
                temp.append(count)
                id_mapping.update({token:count})
                id2word.append(token)
                count += 1
        if max(map(len,temp_facts)) > 55:
            continue
        #temp.append(1)
#        if len(temp) > q_max_len:
#            continue
        #document
        d_max_len = max(map(len,temp_facts)+[d_max_len])
        train_facts.append(list(pad_array(temp_facts,d_max_len))+[[0]*d_max_len]*(d_max_sent-len(temp_facts)))
        train_d_length.append(len(temp_facts))
        d_max_sent = max(d_max_sent,len(temp_facts))
        train_d_sent_len.append(map(len,temp_facts)+[0]*(d_max_sent-len(temp_facts)))
        #question
        train_question.append(temp)
        train_q_length.append(len(temp))
        q_max_len = max(q_max_len,len(temp))
        #ans
        train_a_length.append(len(ans_temp))
        train_answer.append(ans_temp)
        #query_id
        train_id.append(line['query_id'])
        #query_type
        train_type.append(mapping[line['query_type']])
#dev_facts, dev_question, dev_answer = [], [], []
dev_d_length,dev_q_length ,dev_a_length,dev_d_sent_len = [],[],[],[]
dev_id, dev_type = [], []
with open('dev_v1.1.json') as f:
    for line in f:
        number_dict = {}
        line = json.loads(line)
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
                        number_dict.update({token:"tag2number_"+str(len(number_dict))})
                        token = "tag2number_" + str(len(number_dict)-1)
            if token in id_mapping:
                ans_temp.append(id_mapping.get(token))
            else:
                ans_temp.append(count)
                id_mapping.update({token:count})
                id2word.append(token)
                count += 1
        ans_temp.append(1) #EOS
        if len(ans_temp) > a_max_len - 1:
            ans_temp = ans_temp[:a_max_len - 1]
            ans_temp.append(1) #EOS
            #continue
        #document
        temp_facts = []
        is_selected = False
        for passages in line['passages']:
            if passages['is_selected']  == 1:
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
                                    number_dict.update({token:"tag2number_"+str(len(number_dict))})
                                    token = "tag2number_" + str(len(number_dict)-1)
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
                    temp_facts.append(temp)
                is_selected = True
        if is_selected == False:#or len(temp_facts) > d_max_len: #temp_facts No info
            continue
        #if len(ans_temp) > a_max_len:
        #    continue
        #train_question
        temp = []
        for token in word_tokenize(remove_urls(line['query'].lower())):
            token = regex.sub(u'',token)
            if token == u'':
                continue
            if token.isdigit():
                if args.replace == 1:
                    temp.append(tag2number)
                    continue
                elif args.replace == 2:
                    if token in number_dict:
                        token = number_dict.get(token)
                    else:
                        number_dict.update({token:"tag2number_"+str(len(number_dict))})
                        token = "tag2number_" + str(len(number_dict)-1)
            if token in id_mapping:
                temp.append(id_mapping.get(token))
            else:
                temp.append(count)
                id_mapping.update({token:count})
                id2word.append(token)
                count += 1
        if len(temp) > q_max_len:
            temp = temp[:q_max_len]
            #continue
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
        dev_question.append(temp)
        dev_q_length.append(len(temp))
        q_max_len = max(q_max_len,len(temp))
        #ans
        dev_a_length.append(len(ans_temp))
        dev_answer.append(ans_temp)
        #query_id
        dev_id.append(line['query_id'])
        #query_type
        dev_type.append(mapping[line['query_type']])

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

num_symbol = len(id2word)
rnn_size = args.rnn_size
layer = 1
embedding_size = args.embedding_size
index_list = np.array(range(len(train_answer)))
model = qaseq.Model(d_max_length=d_max_len,q_max_length=q_max_len,a_max_length=a_max_len,num_symbol=num_symbol,rnn_size=rnn_size,layer=layer,embedding_size=embedding_size,d_max_sent=d_max_sent)
model.build_model()
print d_max_len,q_max_len,a_max_len,num_symbol,rnn_size,layer,embedding_size,d_max_sent
batch_size = args.batch_size
nb_batch = len(index_list)/batch_size
print 'train vrae'
for epoch in range(5):
    avg = 0.
    np.random.shuffle(index_list)
    # variantional recurrent autoencoder
    batch_size = args.batch_size
    for num in range(nb_batch):
        opti = model.vae_update 
        loss = model.output_net['vae_loss']
        _, cost = model.sess.run([opti,loss],
                feed_dict={
                    model.input_net['d']:train_facts[index_list[num*batch_size:num*batch_size+batch_size]],
                    model.input_net['d_mask']:train_d_sent_len[index_list[num*batch_size:num*batch_size+batch_size]],
                    model.input_net['d_sent_mask']:train_d_length[index_list[num*batch_size:num*batch_size+batch_size]],
                    model.input_net['q']:train_question[index_list[num*batch_size:num*batch_size+batch_size]],
                    model.input_net['q_mask']:train_q_length[index_list[num*batch_size:num*batch_size+batch_size]],
                    model.input_net['a']:train_answer[index_list[num*batch_size:num*batch_size+batch_size]],
                    model.input_net['a_mask']:train_a_length[index_list[num*batch_size:num*batch_size+batch_size]],
                    model.input_net['drop']:0.5})
        avg += cost
        sys.stdout.write(str(epoch)+"\t traininig loss "+str(avg/(num+1))+"\r")
        sys.stdout.flush()
    sys.stdout.write(str(epoch)+"\t traininig loss "+str(avg/nb_batch)+"\t"+"\n")
print 'DMN training'
for epoch in range(150):
    avg = 0.
    np.random.shuffle(index_list)
    ### train on msmarco
    batch_size = args.batch_size
    for num in range(nb_batch):
        opti = model.update
        loss = model.output_net['loss']
        _, cost = model.sess.run([opti,loss],
                feed_dict={
                    model.input_net['d']:train_facts[index_list[num*batch_size:num*batch_size+batch_size]],
                    model.input_net['d_mask']:train_d_sent_len[index_list[num*batch_size:num*batch_size+batch_size]],
                    model.input_net['d_sent_mask']:train_d_length[index_list[num*batch_size:num*batch_size+batch_size]],
                    model.input_net['q']:train_question[index_list[num*batch_size:num*batch_size+batch_size]],
                    model.input_net['q_mask']:train_q_length[index_list[num*batch_size:num*batch_size+batch_size]],
                    model.input_net['a']:train_answer[index_list[num*batch_size:num*batch_size+batch_size]],
                    model.input_net['a_mask']:train_a_length[index_list[num*batch_size:num*batch_size+batch_size]],
                    model.input_net['drop']:0.5})
        avg += cost
        sys.stdout.write(str(epoch)+"\t traininig loss "+str(avg/(num+1))+"\r")
        sys.stdout.flush()
    sys.stdout.write(str(epoch)+"\t traininig loss "+str(avg/nb_batch)+"\t"+"\n")
    ###test on msmarco
    loss = 0.
    candidate = open('/home_local/poyuwu/QA/result/'+ args.device+'/cand_'+str(epoch)+".json",'w')
    if epoch == 0:
        reference = open('/home_local/poyuwu/QA/result/'+ args.device +'/ref_'+str(epoch)+".json",'w')
    #f_attn = open('/home_local/poyuwu/QA/result/'+args.device+'/weight'+str(epoch),'ab')
    batch_size = 128
    for i in range(len(dev_facts)/batch_size +1):
        start = i*batch_size
        end = i*batch_size + batch_size
        if i == len(dev_facts)/batch_size:
            end = len(dev_facts)
        cost, predict = model.sess.run([model.output_net['test_loss'], model.predict],
                    feed_dict={
                        model.input_net['d']:dev_facts[start:end],
                        model.input_net['d_mask']:dev_d_sent_len[start:end],#dev_d_length[start:end],
                        model.input_net['d_sent_mask']:dev_d_length[start:end],
                        model.input_net['q']:dev_question[start:end],
                        model.input_net['q_mask']:dev_q_length[start:end],
                        model.input_net['a']:dev_answer[start:end],
                        model.input_net['a_mask']:dev_a_length[start:end],
                        model.input_net['drop']:1})
        #np.savetxt(f_attn,weight.reshape(end-start,2*d_max_sent))
        loss += cost * batch_size
        #sys.stdout.write(str(epoch)+"\t ms loss:"+str(np.mean(cost))+"\r")
        #sys.stdout.flush()
        for j in range(end-start):
            candidate.write('{"query_id": '+str(dev_id[start+j]) +',"query_type":'+str(dev_type[start+j])+', "answers": ["')
            if epoch == 0:
                reference.write('{"query_id": '+str(dev_id[start+j])+',"query_type":'+str(dev_type[start+j])+ ', "query": "')
                for word in dev_question[start:end][j]:
                    if word == 0:
                        break
                    reference.write(id2word[word].encode('utf8')+' ')
                reference.write('" ,"answers": ["')
            for k in range(a_max_len-1):
                word = predict[k][j]
                if word == 1:
                    break
                candidate.write(id2word[word].encode('utf8')+" ")
                #f_out.write(id2word[word].encode('utf8')+" ")
            #f_out.write('\n')
            #f_out.write("acc: ")
            candidate.write('"]}\n')
            if epoch!=0:
                continue
            for word in dev_answer[start:end][j]:
                if word == 3:
                    continue
                if word == 1:
                    break
                reference.write(id2word[word].encode('utf8')+" ")
            reference.write('"]}\n')
                #f_out.write(id2word[word].encode('utf8')+" ")
            #f_out.write('\n')
    reference.close()
    candidate.close()
    sys.stdout.write(str(epoch)+"\t ms loss:"+str(loss/len(dev_facts))+"\n")
    #print "lr:",model.sess.run(model.learning_rate_decay_op)

import tensorflow as tf
import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

def norm(tensor):#normalzie last line
    return tensor/(tf.sqrt(tf.reduce_sum(tf.square(tensor),-1,keep_dims=True))+1e-12)
def cos(tensor1,tensor2):#by last dimension
    return tf.reduce_sum(tf.mul(norm(tensor1),norm(tensor2)),axis=-1)
def last_relevant(output, length):
    batch_size = tf.shape(output)[0]
    max_length = tf.shape(output)[1]
    out_size = int(output.get_shape()[2])
    index = tf.range(0, batch_size) * max_length + (length - 1)
    flat = tf.reshape(output, [-1, out_size])
    relevant = tf.gather(flat, index)
    return relevant
def softmax(inputs,mask):
    inputs =  tf.exp(inputs) * mask
    sigma = tf.reduce_sum(inputs,axis=1,keep_dims=True)
    return inputs/(sigma+1e-15)
class Model():
    def __init__(self,d_max_length=100,q_max_length=27,a_max_length=27,rnn_size=64,embedding_size=300,num_symbol=10000,atten_sim='nn',layer=2,atten_decoder=False,ss=False,d_max_sent=29):
        tf.reset_default_graph()
        self.d_max_sent = d_max_sent
        self.d_max_length = d_max_length
        self.q_max_length = q_max_length
        self.a_max_length = a_max_length
        self.rnn_size = rnn_size
        self.lr = 1e-2
        self.input_net = {}
        self.output_net = {}
        self.input_net['drop'] = tf.placeholder(tf.float32,[])
        self.cell = tf.nn.rnn_cell.GRUCell(self.rnn_size)
        self.cell = tf.nn.rnn_cell.DropoutWrapper(self.cell,self.input_net['drop'])
        self.fw_cell = tf.nn.rnn_cell.GRUCell(self.rnn_size/2)
        self.bw_cell = tf.nn.rnn_cell.GRUCell(self.rnn_size/2)
        self.result_cell = tf.nn.rnn_cell.GRUCell(self.rnn_size)
        self.result_cell = tf.nn.rnn_cell.DropoutWrapper(self.result_cell,self.input_net['drop'])
        #self.decoder_cell = tf.nn.rnn_cell.GRUCell(self.rnn_size)
        #self.decoder_cell = tf.nn.rnn_cell.DropoutWrapper(self.decoder_cell,self.input_net['drop'])
        self.embedding_size = embedding_size
        self.num_symbol = num_symbol
        self.output_net['loss'] = 0.
        #self.output_net['test_loss'] = 0.
        self.atten_sim = atten_sim
        self.atten_decoder = atten_decoder
        self.num_class = 2
        #self.ss = ss
        self.sess = tf.Session()
    def positional_encoding(self,D,M):
        encoding = np.zeros([D, M])
        for j in range(M):
            for d in range(D):
                encoding[d, j] = (1 - float(j)/M) - (float(d)/D)*(1 - 2.0*j/M)
        return encoding
    def build_model(self,):
        self.input_net['d'] = tf.placeholder(tf.int32,[None,self.d_max_sent,self.d_max_length])
        self.input_net['q'] = tf.placeholder(tf.int32,[None,self.q_max_length])
        #self.input_net['a'] = tf.placeholder(tf.int32,[None,self.a_max_length])
        self.input_net['d_mask'] = tf.placeholder(tf.int32,[None,self.d_max_sent])
        self.input_net['q_mask'] = tf.placeholder(tf.int32,[None])
        #self.input_net['a_mask'] = tf.placeholder(tf.int32,[None])
        self.input_net['labels'] = tf.placeholder(tf.float32,[None,self.num_class])
        self.input_net['d_sent_mask'] = tf.placeholder(tf.int32,[None])
        self.W = tf.Variable(tf.random_uniform([self.num_symbol,self.embedding_size],-3**0.5,3**0.5),name="embedding")
        inner = self.rnn_size
        w1 = tf.get_variable("w1",[self.rnn_size*4,inner],initializer=tf.contrib.layers.xavier_initializer())
        b1 = tf.get_variable("b1",[1,inner],initializer=tf.contrib.layers.xavier_initializer())
        w2 = tf.get_variable("w2",[inner,1],initializer=tf.contrib.layers.xavier_initializer())
        b2 = tf.get_variable("b2",[1,1],initializer=tf.contrib.layers.xavier_initializer())
        classifer_w = tf.get_variable("classifer_w",[inner,2],initializer=tf.contrib.layers.xavier_initializer())
        classifer_b = tf.get_variable("classifer_b",[2],initializer=tf.contrib.layers.xavier_initializer())
        #w4 = tf.get_variable("w4",[32,self.num_class],initializer=tf.contrib.layers.xavier_initializer())
        #b4 = tf.get_variable("b4",[self.num_class],initializer=tf.contrib.layers.xavier_initializer())
        #document
        reader_out = []
        #1
        '''
        # East to overfit ?
        for i in range(self.d_max_sent):
            with tf.variable_scope('reader') as vs:
                if i>0: vs.reuse_variables()
                temp, _ = tf.nn.dynamic_rnn(self.cell,
                                            tf.nn.embedding_lookup(self.W,self.input_net['d'][:,i]),
                                            sequence_length=self.input_net['d_mask'][:,i],
                                            dtype=tf.float32)
                reader_out.append(last_relevant(temp,self.input_net['d_mask'][:,i]))
        '''
        #2 Position Encoding
        ps = self.positional_encoding(self.d_max_length,self.embedding_size)
        input_embed = [tf.nn.embedding_lookup(self.W,sent) for sent in tf.unpack(self.input_net['d'],axis=1)]
        #len = self.d_max_sent
        d_mask = [tf.sequence_mask(mask,self.d_max_length,dtype=tf.float32) for mask in tf.unpack(self.input_net['d_mask'],axis=1)]
        reader_out = [tf.reduce_sum(ps * input_embed[i] * tf.expand_dims(d_mask[i],axis=2) ,axis=1)for i in range(self.d_max_sent)]
        
        #question
        #1
        with tf.variable_scope('reader2') as vs:
            #vs.reuse_variables()
            _, last_q = tf.nn.dynamic_rnn(self.cell,
                                        tf.nn.embedding_lookup(self.W,self.input_net['q']),
                                        sequence_length=self.input_net['q_mask'],
                                        dtype=tf.float32)
            #last_q = last_relevant(temp,self.input_net['q_mask'])
        ##2
        #ps = self.positional_encoding(self.q_max_length,self.embedding_size)
        #q_embed = tf.nn.embedding_lookup(self.W, self.input_net['q'])
        #q_mask = tf.sequence_mask(self.input_net['q_mask'], self.q_max_length, dtype=tf.float32)
        #last_q = tf.reduce_sum(ps * q_embed * tf.expand_dims(q_mask,axis=2) ,axis=1)
        
        # paragraph
        with tf.variable_scope('paragraph'):
            temp, _ = tf.nn.bidirectional_dynamic_rnn(
                    self.fw_cell,
                    self.bw_cell,
                    tf.pack(reader_out,axis=1),
                    sequence_length=self.input_net['d_sent_mask'],
                    dtype=tf.float32)
        reader_out = tf.concat_v2(temp,2)
        self.m_prev = last_q
        for hop in range(2):
            # get content by attention
            self.attention_weight = []
            for i in range(self.d_max_sent):
                vec1 = reader_out[:,i] * last_q
                vec2 = tf.abs(reader_out[:,i] - last_q)
                vec3 = reader_out[:,i] * self.m_prev
                vec4 = tf.abs(reader_out[:,i] - self.m_prev)
                vec = tf.concat(1,[vec1,vec2,vec3,vec4])
                self.attention_weight.append(tf.matmul(tf.tanh(tf.matmul(vec,w1) +b1),w2)+b2)
            self.attention_weight = tf.reshape(tf.pack(self.attention_weight,axis=1),[-1,self.d_max_sent])
            self.attention_weight = softmax(
                    self.attention_weight,
                    tf.sequence_mask(self.input_net['d_sent_mask'],self.d_max_sent,dtype=tf.float32))
            self.context_vec = tf.reduce_sum(reader_out * tf.expand_dims(self.attention_weight,axis=2),axis=1)
            # Update Memory
            ## 1 ReLU 
            #self.m_prev = tf.nn.relu(tf.matmul(tf.concat(1,[self.m_prev,self.context_vec,last_q]),mem_update)+bias)
            ## 2 tied model
            with tf.variable_scope('mem_update') as vs:
                if hop>0: vs.reuse_variables()
                output, self.m_prev = tf.nn.dynamic_rnn(
                    self.result_cell,
                    tf.expand_dims(self.context_vec,axis=1),
                    initial_state= self.m_prev)
        
        input_feature = output[:,0,:]#tf.concat(1, [reader_out, last_q] )
        self.predict = tf.matmul(input_feature,classifer_w) + classifer_b#),w4 )+ b4 
        self.output_net['loss'] = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self.predict,
                                                                                labels = self.input_net['labels']) )
                                                                                        #pos_weight=5))
        self.predict = tf.argmax(self.predict,1)
        actual = tf.argmax(self.input_net['labels'],axis=1)
        self.tp = tf.count_nonzero(self.predict * actual)
        self.tn = tf.count_nonzero((self.predict-1)*(actual-1))
        self.fp = tf.count_nonzero(self.predict*(actual-1))
        self.fn = tf.count_nonzero((self.predict-1)*actual)
        #self.acc = tf.reduce_sum(tf.cast(tf.equal(self.predict,actual),tf.float32))
        # Update
        #self.opti = tf.train.GradientDescentOptimizer(0.01)#.minimize(self.output_net['loss'])
        l2_loss = tf.nn.l2_loss(w1)+ tf.nn.l2_loss(w2) + tf.nn.l2_loss(self.W) + tf.nn.l2_loss(classifer_w)# + tf.nn.l2_loss(w4)
        self.update = tf.train.MomentumOptimizer(0.01,momentum=0.90).minimize(self.output_net['loss']+0.0002*l2_loss)
        #grads_and_vars = self.opti.compute_gradients(self.output_net['loss']+0.0005*l2_loss)
        #capped_grads_and_vars = [ (tf.clip_by_value(gv[0], -0.1, 0.1), gv[1]) for gv in grads_and_vars ]
        #self.update = self.opti.apply_gradients(capped_grads_and_vars)

        init = tf.global_variables_initializer()#
        self.sess.run(init)
    #def transform(self,inputs,output_projection):#,outputs):
    #    return [tf.matmul(input,output_projection[0])+output_projection[1] for input in inputs]

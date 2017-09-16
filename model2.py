import tensorflow as tf
import os
import numpy as np

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
    def __init__(self,d_max_length=100,q_max_length=27,a_max_length=27,rnn_size=64,embedding_size=300,num_symbol=10000,atten_sim='nn',layer=2,atten_decoder=False,encode_type='cnn',d_max_sent=29):
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
        self.fw_cell = tf.nn.rnn_cell.GRUCell(self.rnn_size)
        self.bw_cell = tf.nn.rnn_cell.GRUCell(self.rnn_size)
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
        self.l2_loss = tf.constant(0.0)
        self.sess = tf.Session()
        self.encode_type = encode_type
    def positional_encoding(self, D, M):
        encoding = np.zeros([D, M])
        for j in range(M):
            for d in range(D):
                encoding[d, j] = (1 - float(j+1)/M) - (float(d+1)/D)*(1 - 2.0*(j+1)/M)
        return np.transpose(encoding)
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
        self.l2_loss += tf.nn.l2_loss(self.W)
        inner = self.rnn_size
        w1 = tf.get_variable("w1",[self.rnn_size*4,inner])#,initializer=tf.contrib.layers.xavier_initializer())
        #self.l2_loss += tf.nn.l2_loss(w1)
        b1 = tf.get_variable("b1",[1,inner])#,initializer=tf.contrib.layers.xavier_initializer())
        w2 = tf.get_variable("w2",[inner,1])#,initializer=tf.contrib.layers.xavier_initializer())
        #self.l2_loss += tf.nn.l2_loss(w2)
        b2 = tf.get_variable("b2",[1,1])#,initializer=tf.contrib.layers.xavier_initializer())
        classifer_w = tf.get_variable("classifer_w",[inner*2,2])#,initializer=tf.contrib.layers.xavier_initializer())
        self.l2_loss += tf.nn.l2_loss(classifer_w)
        classifer_b = tf.get_variable("classifer_b",[2])#,initializer=tf.contrib.layers.xavier_initializer())
        #w4 = tf.get_variable("w4",[33,self.num_class],initializer=tf.contrib.layers.xavier_initializer())
        #b4 = tf.get_variable("b4",[self.num_class],initializer=tf.contrib.layers.xavier_initializer())
        #document
        reader_out = []
        #1
        '''
        # East to overfit?
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
        ps = self.positional_encoding(self.embedding_size, self.d_max_length)
        input_embed = [tf.nn.embedding_lookup(self.W,sent) for sent in tf.unpack(self.input_net['d'],axis=1)]
        #len = self.d_max_sent
        d_mask = [tf.sequence_mask(mask,self.d_max_length,dtype=tf.float32) for mask in tf.unpack(self.input_net['d_mask'],axis=1)]
        reader_out = [tf.reduce_sum(ps * input_embed[i] * tf.expand_dims(d_mask[i],axis=2) ,axis=1) for i in range(self.d_max_sent)]
        # reader_out is a d_max_sent list of 2D Tensors, shape(None, embedding_size)
        #question
        #1
        with tf.variable_scope('reader2') as vs:
            #vs.reuse_variables()
            temp, _ = tf.nn.dynamic_rnn(self.cell,
                                        tf.nn.embedding_lookup(self.W,self.input_net['q']),
                                        sequence_length=self.input_net['q_mask'],
                                        dtype=tf.float32)
            last_q = last_relevant(temp,self.input_net['q_mask'])
        ##2
        #ps = self.positional_encoding(self.q_max_length,self.embedding_size)
        #q_embed = tf.nn.embedding_lookup(self.W, self.input_net['q'])
        #q_mask = tf.sequence_mask(self.input_net['q_mask'], self.q_max_length, dtype=tf.float32)
        #last_q = tf.reduce_sum(ps * q_embed * tf.expand_dims(q_mask,axis=2) ,axis=1)
        
        # paragraph vector
        # 1. Using RNN
        if self.encode_type == "rnn":
            with tf.variable_scope('paragraph'):
                temp, _ = tf.nn.bidirectional_dynamic_rnn(
                        self.fw_cell,
                        self.bw_cell,
                        tf.pack(reader_out,axis=1),
                        sequence_length=self.input_net['d_sent_mask'],
                        dtype=tf.float32)
            reader_out = tf.reduce_sum(tf.stack(temp),axis=0)
            input_feature = last_relevant(reader_out, self.input_net['d_sent_mask'])
        elif self.encode_type == "cnn":
            # 2. Using CNN with MaxPooling
            reader_out = tf.expand_dims(tf.pack(reader_out, axis=1), -1) 
            filter_sizes = [3]
            pooled_output = []
            for i, filter_size in enumerate(filter_sizes):
                with tf.name_scope('conv-maxpool-%s' % filter_size):
                    filter_shape = [filter_size, self.embedding_size, 1, self.rnn_size] # rnn_size => # of filter
                    conv_W = tf.Variable(tf.truncated_normal(filter_shape, stddev = 0.1), name="conv_W")
                    conv_b = tf.Variable(tf.constant(0.1, shape=[self.rnn_size], name='conv_b'))
                    conv = tf.nn.conv2d(reader_out,
                                        conv_W,
                                        strides=[1, 1, 1, 1],
                                        padding="VALID",
                                        name="conv")
                    h = tf.nn.relu(conv + conv_b ,name="relu")
                    pooled = tf.nn.max_pool(
                            h,
                            ksize=[1, self.d_max_sent - filter_size + 1, 1, 1],
                            strides=[1, 1, 1, 1],
                            padding="VALID",
                            name="pooled")
                    pooled_output.append(pooled)
            num_filters_total = self.rnn_size * len(filter_sizes)
            h_pool = tf.concat(3, pooled_output)
            input_feature = tf.reshape(h_pool, [-1, num_filters_total])

        #input_feature = output[:,0,:]#tf.concat(1, [reader_out, last_q] )
        self.prob = tf.matmul(tf.concat(1,[input_feature,last_q]), classifer_w) + classifer_b#),w4 )+ b4 
        self.output_net['loss'] = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self.prob,
                                                                                labels = self.input_net['labels']) )
                                                                                        #pos_weight=5))
        self.predict = tf.argmax(self.prob, 1)
        actual = tf.argmax(self.input_net['labels'],axis=1)
        self.tp = tf.count_nonzero(self.predict * actual)
        self.tn = tf.count_nonzero((self.predict-1)*(actual-1))
        self.fp = tf.count_nonzero(self.predict*(actual-1))
        self.fn = tf.count_nonzero((self.predict-1)*actual)
        #self.acc = tf.reduce_sum(tf.cast(tf.equal(self.predict,actual),tf.float32))
        # Update
        #self.opti = tf.train.GradientDescentOptimizer(0.01)#.minimize(self.output_net['loss'])
        self.update = tf.train.AdamOptimizer(0.005).minimize(self.output_net['loss'])
        #grads_and_vars = self.opti.compute_gradients(self.output_net['loss']+0.0005*l2_loss)
        #capped_grads_and_vars = [ (tf.clip_by_value(gv[0], -0.1, 0.1), gv[1]) for gv in grads_and_vars ]
        #self.update = self.opti.apply_gradients(capped_grads_and_vars)

        init = tf.global_variables_initializer()#
        self.sess.run(init)
    #def transform(self,inputs,output_projection):#,outputs):
    #    return [tf.matmul(input,output_projection[0])+output_projection[1] for input in inputs]

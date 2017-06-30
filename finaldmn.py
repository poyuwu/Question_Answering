import tensorflow as tf
import numpy as np
import seq2seq

#def norm(tensor):#normalzie last line
#    """ Normalize tensor alone last dimension. Be equal to tf.nn.l2_normalize
#        
#        Args:
#          tensor: a tensorflow tensor
#        
#        Returns: 
#          return a normalized tensor, 1e-12 for avoiding zero division
#    """
#    return tensor/(tf.sqrt(tf.reduce_sum(tf.square(tensor),-1,keep_dims=True))+1e-12)
#def cos(tensor1,tensor2):#by last dimension
#    return tf.reduce_sum(tf.mul(norm(tensor1),norm(tensor2)),axis=-1)
def last_relevant(output, length):
    """ Return last time step of RNNs
        Args:
          output: RNNs output. 3D Tensors, [None, sequence length, rnn cell size]
          length: sequence masking. 1D batch-sized int32 Tensors
        
        Returns:
          last available time step of RNNs.
    """
    batch_size = tf.shape(output)[0]
    max_length = tf.shape(output)[1]
    out_size = int(output.get_shape()[2])
    index = tf.range(0, batch_size) * max_length + (length - 1)
    flat = tf.reshape(output, [-1, out_size])
    relevant = tf.gather(flat, index)
    return relevant
def softmax(inputs,mask):
    """ Calculate softmax with sequece mask.
        
        Args:
          inputs: 2D tensor.
          mask: A 2D mask tensor. Corresponding to inputs sequence length, similiar to rnn sequence length.
                ex: [[1.,1.,0.],[1.,0.,0.]]. it means two sentence with 2 and 1 word, respectively.
        Returns:
            2D tensor after softmax.
    """
    inputs =  tf.exp(inputs) * mask
    sigma = tf.reduce_sum(inputs,axis=1,keep_dims=True)
    return inputs/(sigma+1e-12) 
class Model():
    def __init__(self,d_max_length=100,q_max_length=27,a_max_length=27,rnn_size=64,embedding_size=300,num_symbol=10000,layer=2,d_max_sent=29):
        tf.reset_default_graph()
        self.d_max_sent = d_max_sent
        self.d_max_length = d_max_length
        self.q_max_length = q_max_length
        self.a_max_length = a_max_length
        self.rnn_size = rnn_size
        #self.lr = 1e-2
        self.input_net = {}
        self.output_net = {}
        self.input_net['drop'] = tf.placeholder(tf.float32,[])
        self.latent_dim = 64
        self.cell = tf.nn.rnn_cell.GRUCell(self.rnn_size)
        self.cell = tf.nn.rnn_cell.DropoutWrapper(self.cell,self.input_net['drop'])
        self.a_cell = tf.nn.rnn_cell.GRUCell(self.rnn_size)
        self.a_cell = tf.nn.rnn_cell.DropoutWrapper(self.a_cell,self.input_net['drop'])
        self.fw_cell = tf.nn.rnn_cell.GRUCell(self.rnn_size/2)
        self.bw_cell = tf.nn.rnn_cell.GRUCell(self.rnn_size/2)
        self.result_cell = tf.nn.rnn_cell.GRUCell(self.rnn_size)
        self.result_cell = tf.nn.rnn_cell.DropoutWrapper(self.result_cell,self.input_net['drop'])
        self.decoder_cell = tf.nn.rnn_cell.GRUCell(self.rnn_size)
        self.decoder_cell = tf.nn.rnn_cell.DropoutWrapper(self.decoder_cell,self.input_net['drop'])
        self.embedding_size = embedding_size
        self.num_symbol = num_symbol
        #self.output_net['loss'] = 0.
        #self.output_net['test_loss'] = 0.
        #self.atten_sim = atten_sim
        #self.atten_decoder = atten_decoder
        #self.ss = ss
        self.sess = tf.Session()
        self.l2_loss1 = tf.constant(0.0)
        self.l2_loss2 = tf.constant(0.0)
    def positional_encoding(self,D,M):
        encoding = np.zeros([D, M])
        for j in range(M):
            for d in range(D):
                encoding[d, j] = (1 - float(j+1)/M) - (float(d+1)/D)*(1 - 2.0*(j+1)/M)
        return np.transpose(encoding)
    def build_model(self,):
        self.input_net['d'] = tf.placeholder(tf.int32,[None,self.d_max_sent,self.d_max_length])
        self.input_net['q'] = tf.placeholder(tf.int32,[None,self.q_max_length])
        self.input_net['a'] = tf.placeholder(tf.int32,[None,self.a_max_length])
        self.input_net['d_mask'] = tf.placeholder(tf.int32,[None,self.d_max_sent])
        self.input_net['q_mask'] = tf.placeholder(tf.int32,[None])
        self.input_net['a_mask'] = tf.placeholder(tf.int32,[None])
        self.input_net['d_sent_mask'] = tf.placeholder(tf.int32,[None])
        self.encoder_W = tf.Variable(tf.random_uniform([self.num_symbol,self.embedding_size],-3**0.5,3**0.5),name="embedding")
        self.decoder_W = tf.Variable(tf.random_uniform([self.num_symbol,self.embedding_size],-3**0.5,3**0.5),name="embedding_decoder")
        self.l2_loss1 += tf.nn.l2_loss(self.encoder_W)
        self.l2_loss2 += tf.nn.l2_loss(self.encoder_W)
        self.l2_loss1 += tf.nn.l2_loss(self.decoder_W)
        self.l2_loss2 += tf.nn.l2_loss(self.decoder_W)
        inner = self.rnn_size
        w1 = tf.get_variable("w1",[self.rnn_size*4,inner],initializer=tf.contrib.layers.xavier_initializer())
        b1 = tf.get_variable("b1",[1,inner],initializer=tf.contrib.layers.xavier_initializer())
        w2 = tf.get_variable("w2",[inner,1],initializer=tf.contrib.layers.xavier_initializer())
        b2 = tf.get_variable("b2",[1,1],initializer=tf.contrib.layers.xavier_initializer())
        # create variantional recurrent autoencoder 
        a_embed = tf.nn.embedding_lookup(self.encoder_W,self.input_net['a'][:,1:])
        _, a_enc_state = tf.nn.dynamic_rnn(
                            self.a_cell,
                            a_embed,
                            sequence_length = self.input_net['a_mask']-2,
                            dtype = tf.float32)
        ## encoder to latent
        with tf.variable_scope('encode_to_latent'):
            w_enc_latent = tf.get_variable("w_enc_latent",[self.rnn_size,2*self.latent_dim],dtype=tf.float32)
            self.l2_loss1 += tf.nn.l2_loss(w_enc_latent)
            b_enc_latent = tf.get_variable("b_enc_latent",[2*self.latent_dim],dtype=tf.float32,initializer=tf.zeros_initializer)
            # encoder to mean and variance
            self.mu_enc, self.sig_enc = tf.split(1, 2, seq2seq.prelu(tf.matmul(a_enc_state,w_enc_latent)+b_enc_latent))
            # sample latent space
            z, self.kl_obj, self.kl_cost = seq2seq.sample(self.mu_enc, self.sig_enc, self.latent_dim, kl_min=4)
        with tf.variable_scope('latent_to_decoder'):
            W_z = tf.get_variable("W_z",[self.latent_dim,self.rnn_size],initializer=tf.contrib.layers.xavier_initializer())
            bias_z = tf.get_variable("bias_z",[self.rnn_size],initializer=tf.zeros_initializer)
            self.l2_loss1 += tf.nn.l2_loss(W_z)
            vae_decoder = seq2seq.prelu(tf.matmul(z,W_z)+bias_z)

        ## read document to sentence vector
        #1
        '''
        # East to overfit
        reader_out = []
        for i in range(self.d_max_sent):
            with tf.variable_scope('reader') as vs:
                if i>0: vs.reuse_variables()
                temp, _ = tf.nn.dynamic_rnn(self.cell,
                                            tf.nn.embedding_lookup(self.encoder_W,self.input_net['d'][:,i]),
                                            sequence_length=self.input_net['d_mask'][:,i],
                                            dtype=tf.float32)
                reader_out.append(last_relevant(temp,self.input_net['d_mask'][:,i]))
        '''
        #2 Position Encoding
        ps = self.positional_encoding(self.embedding_size,self.d_max_length)
        input_embed = [tf.nn.embedding_lookup(self.encoder_W,sent) for sent in tf.unpack(self.input_net['d'],axis=1)]
        #len = self.d_max_sent
        d_mask = [tf.sequence_mask(mask,self.d_max_length,dtype=tf.float32) for mask in tf.unpack(self.input_net['d_mask'],axis=1)]
        reader_out = [tf.reduce_sum(ps * input_embed[i] * tf.expand_dims(d_mask[i],axis=2) ,axis=1)for i in range(self.d_max_sent)]
        
        #question
        #1
        with tf.variable_scope('Question_reader') as vs:
            #vs.reuse_variables()
            _ ,last_q = tf.nn.dynamic_rnn(self.cell,
                                        tf.nn.embedding_lookup(self.encoder_W,self.input_net['q']),
                                        sequence_length=self.input_net['q_mask'],
                                        dtype=tf.float32)
            #last_q = last_relevant(temp,self.input_net['q_mask'])
        ##2
        #ps = self.positional_encoding(self.q_max_length,self.embedding_size)
        #q_embed = tf.nn.embedding_lookup(self.encoder_W, self.input_net['q'])
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
                _, self.m_prev = tf.nn.dynamic_rnn(
                    self.result_cell,
                    tf.expand_dims(self.context_vec,axis=1),
                    initial_state= self.m_prev)
            #enc_out, enc_state = tf.nn.dynamic_rnn(
            #        self.result_cell,
            #        temp,
            #        sequence_length=self.input_net['d_sent_mask'],
            #        dtype=tf.float32)
        # output projection and sampled loss function
        w_t = tf.get_variable("proj_w", [self.num_symbol, self.rnn_size], dtype=tf.float32)
        w = tf.transpose(w_t)
        b = tf.get_variable("proj_b", [self.num_symbol])
        self.output_projection = (w, b)
        def sampled_loss(inputs, labels):
            labels = tf.reshape(labels, [-1, 1]) 
            local_w_t = tf.cast(w_t, tf.float32)
            local_b = tf.cast(b, tf.float32)
            local_inputs = tf.cast(inputs, tf.float32)
            return tf.cast(
                tf.nn.sampled_softmax_loss(local_w_t, local_b, local_inputs, labels,
                                   512, self.num_symbol),tf.float32)
        softmax_loss_function = sampled_loss
        
        # decoder attention weight
        #top_states = [tf.nn.array_ops.reshape(e, [-1, 1, self.result_cell.output_size]) for e in tf.unpack(enc_out,axis=1)]
        #attention_states = tf.nn.array_ops.concat(1, top_states)
        enc_state = self.m_prev
        decode_input = tf.unpack(self.input_net['a'],axis=1)
        #self.decoder_cell = tf.nn.rnn_cell.OutputProjectionWrapper(self.decoder_cell,self.num_symbol)
        # reconstruct vae
        with tf.variable_scope('decoder'):
            self.answer_out, _ = seq2seq.embedding_rnn_decoder(
                    decode_input[:-1],
                    vae_decoder,
                    #attention_states,
                    self.decoder_cell,
                    self.decoder_W,
                    self.num_symbol,
                    self.embedding_size,
                    output_projection=self.output_projection,
                    feed_previous=False)
        # reconstruct testing loss
        with tf.variable_scope('decoder',reuse = True):
            self.answer_test_out, _ = seq2seq.embedding_rnn_decoder(
                    decode_input[:-1],
                    vae_decoder,
                    #attention_states,
                    self.decoder_cell,
                    self.decoder_W,
                    self.num_symbol,
                    self.embedding_size,
                    output_projection=self.output_projection,
                    feed_previous=True)
        # DMN decoder training
        with tf.variable_scope('decoder',reuse=True):
            self.a_out, _ = seq2seq.embedding_rnn_decoder(
                    decode_input[:-1],
                    enc_state,
                    #attention_states,
                    self.decoder_cell,
                    self.decoder_W,
                    self.num_symbol,
                    self.embedding_size,
                    output_projection=self.output_projection,
                    feed_previous=False)
        #self.a_out = [tf.stop_gradient(iteration) for iteration in self.a_out]
        # DMN decoder testing
        with tf.variable_scope('decoder',reuse=True):
            self.a_predict, _ = seq2seq.embedding_rnn_decoder(
                    decode_input[:-1],
                    enc_state,
                    #attention_states,
                    self.decoder_cell,
                    self.decoder_W,
                    self.num_symbol,
                    self.embedding_size,
                    output_projection=self.output_projection,
                    feed_previous=True)
        # loss function
        # remove GO length
        a_mask = tf.sequence_mask(self.input_net['a_mask']-1 , self.a_max_length -1 , dtype=tf.float32)
        a_mask = tf.unpack(a_mask,axis=1)
        self.output_net['vae_loss'] = tf.nn.seq2seq.sequence_loss_by_example(
                                        self.answer_out,
                                        decode_input[1:],
                                        a_mask,
                                        softmax_loss_function = softmax_loss_function)
        self.output_net['loss'] = tf.nn.seq2seq.sequence_loss_by_example(
                                    self.a_out,
                                    decode_input[1:],
                                    a_mask,
                                    softmax_loss_function = softmax_loss_function)
        self.output_net['test_loss'] = tf.nn.seq2seq.sequence_loss_by_example(
                                    self.a_predict,
                                    decode_input[1:],
                                    a_mask,
                                    softmax_loss_function = softmax_loss_function)
        self.output_net['vae_loss'] = tf.reduce_mean(self.output_net['vae_loss'])
        self.output_net['loss'] = tf.reduce_mean(self.output_net['loss'])
        self.output_net['test_loss'] = tf.reduce_mean(self.output_net['test_loss'])

        # To word
        #self.a_out = self.transform(self.a_out,self.output_projection)
        self.a_predict = self.transform(self.a_predict,self.output_projection)
        #self.a_train = [ tf.argmax(word,1) for word in self.a_out ]
        self.predict = [ tf.argmax(word,1) for word in self.a_predict ]
        
        # Update
        #self.opti = tf.train.GradientDescentOptimizer(0.01)#.minimize(self.output_net['loss'])
        self.opti = tf.train.AdamOptimizer(0.001)
        self.vae_update = self.opti.minimize(self.output_net['vae_loss'] + 0.001 * self.l2_loss1 + self.kl_obj)
        self.l2_loss2 += tf.nn.l2_loss(w1)+ tf.nn.l2_loss(w2)
        self.update = tf.train.AdamOptimizer(0.001).minimize(self.output_net['loss'] + 0.001 * self.l2_loss2)
        #self.opti = tf.train.AdamOptimizer(0.01)
        #grads_and_vars = self.opti.compute_gradients(self.output_net['loss'] + 0.001*l2_loss)
        #capped_grads_and_vars = [ (tf.clip_by_value(gv[0], -0.1, 0.1), gv[1]) for gv in grads_and_vars ]
        #self.update = self.opti.apply_gradients(capped_grads_and_vars)

        init = tf.global_variables_initializer()#
        self.sess.run(init)
    def transform(self,inputs,output_projection):#,outputs):
        return [tf.matmul(input,output_projection[0])+output_projection[1] for input in inputs]

import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('TkAgg')
import matplotlib as mpl
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

from tensorflow.examples.tutorials.mnist import input_data

seed = 1234
np.random.seed(seed)
tf.set_random_seed(seed)
import random
random.seed(seed)
import math

mnist_data = input_data.read_data_sets('./data/', one_hot=True)
train_imgs = mnist_data.train.images
test_imgs = mnist_data.test.images
train_labels = mnist_data.train.labels
test_labels = mnist_data.test.labels

def linear(input_node, num_out, scope):
    with tf.variable_scope(scope or 'linear') as scope:
        w = tf.get_variable('w', [input_node.get_shape()[1], num_out], initializer=tf.truncated_normal_initializer())
        b = tf.get_variable('b', [num_out], initializer=tf.constant_initializer())
        return tf.matmul(input_node, w) + b

class FedLearnClient(object):
    def __init__(self, train_imgs, train_labels, client_id, learning_rate=0.0001, batch_size=64):
        self.train_imgs = train_imgs
        self.train_labels = train_labels
        self.client_id = client_id
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.loss_hist = []
        self.err_hist = []
        self.name = format('client_%d' %self.client_id)
        self._create_model(128)
    def _create_model(self, num_hidden):
        with tf.variable_scope(self.name):
            self.x = tf.placeholder(tf.float32, [None, 784], name="x")
            self.y = tf.placeholder(tf.float32, [None, 10], name="y")
            h1 = tf.nn.tanh(linear(self.x, num_hidden, 'h1'))
            self.y_logits = linear(h1, 10, 'h2')
            correct_prediction = tf.equal(tf.argmax(self.y_logits,1), tf.argmax(self.y,1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self.y_logits, labels = self.y))
            self.train_step  = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss) 
    def train(self, session, num_steps):
        for i in range(num_steps):
            for j in range(self.train_imgs.shape[0]//self.batch_size):
                batch_xs = self.train_imgs[j* self.batch_size: (j+1)*self.batch_size]
                batch_ys = self.train_labels[j* self.batch_size: (j+1) * self.batch_size]
                _, cur_acc, cur_loss  = session.run([self.train_step, self.accuracy, self.loss], feed_dict={self.x : batch_xs, self.y: batch_ys})
            self.loss_hist.append(cur_loss)
            self.err_hist.append(1 - cur_acc)         
    def get_loss_hist(self):
        return self.loss_hist
    def get_err_hist(self):
        return self.err_hist                
    def get_weights(self, session):
        return session.run([v for v in tf.trainable_variables() if v.name.startswith(self.name)])    
    def set_weights(self, session, weights):
        my_weights = [v for v in tf.trainable_variables() if v.name.startswith(self.name)]
        for i, v in enumerate(my_weights):
            session.run(v.assign(weights[i]))        
    def get_name(self):
        return self.name 

class FedLearnServer(object):
    def __init__(self, test_imgs, test_labels, server_id):
        self.test_imgs = test_imgs
        self.test_labels = test_labels
        self.loss_hist = []
        self.err_hist = []
        self.server_id = server_id
        self.name = format('server_%d' %self.server_id)
        self._create_model(128)    
    def _create_model(self, num_hidden):
        with tf.variable_scope(self.name):
            self.x = tf.placeholder(tf.float32, [None, 784], name="x")
            self.y = tf.placeholder(tf.float32, [None, 10], name="y")
            h1 = tf.nn.tanh(linear(self.x, num_hidden, 'h1'))
            self.y_logits = linear(h1, 10, 'h2')
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self.y_logits, labels = self.y))
            correct_prediction = tf.equal(tf.argmax(self.y_logits,1), tf.argmax(self.y,1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    def test(self, session):
        cur_acc, cur_loss = session.run([self.accuracy, self.loss], feed_dict={self.x : self.test_imgs, self.y: self.test_labels})
        self.loss_hist.append(cur_loss)
        self.err_hist.append(1-cur_acc)
    def get_loss_hist(self):
        return self.loss_hist
    def get_err_hist(self):
        return self.err_hist
    def get_weights(self, session):
        return session.run([v for v in tf.trainable_variables() if v.name.startswith(self.name)])
    def set_weights(self, session, weights):
        my_weights = [v for v in tf.trainable_variables() if v.name.startswith(self.name)]
        for i, v in enumerate(my_weights):
            session.run(v.assign(weights[i]))
    def get_name(self):
        return self.name

class P2PNetSimulator(object):
    def __init__(self, num_clients, train_imgs, train_labels, test_imgs, test_labels, learning_rate, batch_size):
        self.num_clients = num_clients
        self.train_imgs = train_imgs
        self.train_labels = train_labels
        self.test_imgs = test_imgs
        self.test_labels = test_labels
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.clients = []
        self.servers = []
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.47)
        self.session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)
        self.clients_data, self.clients_share = self._distribute_data(self.num_clients)
        for i in range(num_clients):
            self.clients.append(self._create_client(i, self.clients_data[i][0], self.clients_data[i][1]))
        for i in range(num_clients):
            self.servers.append(self._create_server(i, self.test_imgs, self.test_labels))
        self.session.run(tf.global_variables_initializer())
    def _distribute_data(self, num_clients):
        client_data_size = []
        client_share = []
        for i in range(num_clients):
            client_data_size.append(random.uniform(1, 5))
        for i in range(num_clients - 1):
            client_share.append( (self.train_imgs.shape[0] * client_data_size[i]/sum(client_data_size))//1 )
        client_share.append(self.train_imgs.shape[0] - sum(client_share))       
        client_share_x = range(len(client_share))
        width = 1/2
        plt.bar(client_share_x, client_share, width, color="blue")
        plt.xlabel('Distributed Clients')
        plt.ylabel('Distributed Dataset Size')
        print([int(a) for a in client_share])    
        client_share_1 = np.cumsum([0]+client_share)
        client_share_1 = client_share_1.astype(int)
        training_data = np.concatenate((self.train_imgs, self.train_labels),axis = 1)
        training_data = np.random.permutation(training_data)
        self.train_imgs, self.train_labels, _ = np.split(training_data, [784, 794], axis = 1)
        clients_data = []
        for i in range(num_clients):
            client_imgs = self.train_imgs[client_share_1[i]: client_share_1[i+1]]
            client_labels = self.train_labels[client_share_1[i]: client_share_1[i+1]]
            clients_data.append((client_imgs, client_labels))
        return clients_data, client_share
    def _create_client(self, i, client_x, client_y):
        return FedLearnClient(client_x, client_y, i, self.learning_rate)    
    def _create_server(self, i, server_x, server_y):
        return FedLearnServer(server_x, server_y, i)
    def train(self, num_steps):
        for i, client in enumerate(self.clients):
            client.train(self.session, num_steps)                                          
            trained_param = client.get_weights(self.session)    
            self.servers[i].set_weights(self.session, trained_param)
        print("Training Complete! ")
    def update_weights(self):
        for i in range(self.num_clients):
            if i == 0:
                p2p_params = self.merge_weights([self.num_clients-1, i, i+1])
                self.clients[i].set_weights(self.session, p2p_params)
            elif i == (self.num_clients-1):
                p2p_params = self.merge_weights([i-1, i, 0])
                self.clients[i].set_weights(self.session, p2p_params)
            else:
                p2p_params = self.merge_weights([i-1, i, i+1])
                self.clients[i].set_weights(self.session, p2p_params)
        for i, server in enumerate(self.servers):
            updated_weights = self.clients[i].get_weights(self.session)
            server.set_weights(self.session, updated_weights)
        print("Parameters updating complete")
    def merge_weights(self, working_list):
        client_dataset_weight = []  
        clients_weights = []       
        for i, client in enumerate(self.servers):
            if i in working_list:
                clients_weights.append(client.get_weights(self.session))
                client_dataset_weight.append(self.clients_share[i])
        final_weights = self.get_final_weight(clients_weights, client_dataset_weight)                     
        return final_weights
    def get_final_weight(self, clients_weights, client_dataset_weight):
        clients_num = len(client_dataset_weight)
        client_dataset_weight = np.array(client_dataset_weight)
        clients_weights = np.array(clients_weights)
        for i in range(clients_num):
            clients_weights[i] = clients_weights[i]*client_dataset_weight[i]
        final_weights = np.sum(clients_weights, axis = 0)/ sum(client_dataset_weight)
        return final_weights                                                                
    def test(self):
        for i, server in enumerate(self.servers):
            server.test(self.session)
    def plot_loss_hist(self):
        width = 16
        height = 12
        fig= plt.figure(figsize=(width, height))
        ax = fig.add_subplot(111)
        upate_time = [19, 39, 59, 79, 99, 119, 139, 159, 179, 199]
        for i in range(self.num_clients):
            client_loss = self.servers[i].get_loss_hist()
            plt.plot(upate_time, client_loss, label=self.clients[i].get_name())
        ax.legend()
        plt.xlabel('Iteration')
        plt.ylabel('Testing loss for training-clients')               
    def plot_err_hist(self):
        width = 16
        height = 12
        fig= plt.figure(figsize=(width, height))
        ax = fig.add_subplot(111)
        upate_time = [19, 39, 59, 79, 99, 119, 139, 159, 179, 199]
        for i in range(self.num_clients):
            client_err = self.servers[i].get_err_hist()
            plt.plot(upate_time, client_err, label=self.clients[i].get_name())
        ax.legend()
        plt.xlabel('Iteration')
        plt.ylabel('Error-rate for training-clients')       
    def get_clients_loss_hist(self):
        return [x.get_loss_hist() for x in self.clients]    
    def get_server_loss_hist(self):
        return [x.get_loss_hist() for x in self.servers]    
    def get_server_err_hist(self):
        return [x.get_err_hist() for x in self.servers]    

Iteration = 20
K_param = 10
B_param = 100
E_param = 20
learning_rate = 0.01

simulator = P2PNetSimulator(K_param, train_imgs, train_labels, test_imgs, test_labels, learning_rate, B_param)

for i in range(Iteration):
    simulator.train(E_param)
    simulator.update_weights()
    simulator.test()

result = simulator.get_server_err_hist()
[i[-1] for i in result]

simulator.get_server_err_hist()


import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('TkAgg')
import matplotlib as mpl
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import collections
import os
import zipfile
from six.moves import urllib
from six.moves import xrange    
seed = 1234
np.random.seed(seed)
tf.set_random_seed(seed)
import random
random.seed(seed)
import math

url = 'http://mattmahoney.net/dc/'
def maybe_download(filename, expected_bytes):
    """Download a file if not present, and make sure it's the right size."""
    if not os.path.exists(filename):
        filename, _ = urllib.request.urlretrieve(url + filename, filename)
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified', filename)
    else:
        print(statinfo.st_size)
        raise Exception('Failed to verify ' + filename + '. Can you get to it with a browser?')
    return filename
filename = maybe_download('text8.zip', 31344016)

def read_data(filename):
    """Extract the first file enclosed in a zip file as a list of words"""
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data
words = read_data(filename)
print('Data size', len(words))
vocabulary_size = 50000

def build_dataset(words):
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary

data, count, dictionary, reverse_dictionary = build_dataset(words)
del words    
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])
data_index = 0

def generate_batch(batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1
    buffer = collections.deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    for i in range(batch_size // num_skips):
        target = skip_window    
        targets_to_avoid = [ skip_window ]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    return batch, labels

unigrams = [ c / vocabulary_size for token, c in count ]
batch_size = 128
embedding_size = 128    
skip_window = 2             
num_skips = 1

class FedLearnClient(object):
    def __init__(self, client_id, count):
        self.client_id = client_id
        self.loss_hist = []
        self.name = format('client_%d' %self.client_id)
        self.learning_rate = 0.4
        self.batch_size = 128
        self.embedding_size = 128
        self.vocabulary_size = 50000
        self.skip_window = 2
        self.num_skips = 1
        self.unigrams = [ c / self.vocabulary_size for token, c in count ]   
        self._create_model()          
    def _create_model(self):
        with tf.variable_scope(self.name):
            self.t_inputs = tf.placeholder(tf.int32, shape=[self.batch_size],name="t_inputs")
            self.t_labels = tf.placeholder(tf.int32, shape=[self.batch_size, 1],name="t_labels")
            input_ids = self.t_inputs
            labels = tf.reshape(self.t_labels, [self.batch_size])
            labels_matrix = tf.reshape(tf.cast(labels, dtype=tf.int64), [self.batch_size, 1])
            input_vectors = tf.Variable(tf.random_uniform(
                [self.vocabulary_size, self.embedding_size], -1.0, 1.0), name="input_vectors")
            output_vectors = tf.Variable(tf.random_uniform(
                [self.vocabulary_size, self.embedding_size], -1.0, 1.0), name="output_vectors")
            sampled_ids, _, _ = (tf.nn.fixed_unigram_candidate_sampler(
                    true_classes=labels_matrix, num_true=1, num_sampled=200,
                    unique=True, range_max=vocabulary_size, distortion=0.75,
                    unigrams=unigrams))
            center_vects = tf.nn.embedding_lookup(input_vectors, input_ids)
            context_vects = tf.nn.embedding_lookup(output_vectors, labels)
            sampled_vects = tf.nn.embedding_lookup(output_vectors, sampled_ids)
            incorpus_logits = tf.reduce_sum(tf.multiply(center_vects, context_vects), 1)
            incorpus_probabilities = tf.nn.sigmoid(incorpus_logits)
            sampled_logits = tf.matmul(center_vects, sampled_vects, transpose_b=True)
            outcorpus_probabilities = tf.nn.sigmoid(-sampled_logits)
            outcorpus_loss_perexample = tf.reduce_sum(tf.log(outcorpus_probabilities), 1)
            loss_perexample = - tf.log(incorpus_probabilities) - outcorpus_loss_perexample
            self.loss = tf.reduce_sum(loss_perexample) / self.batch_size
            self.train_step = tf.train.GradientDescentOptimizer(.4).minimize(self.loss)
            norm = tf.sqrt(tf.reduce_sum(tf.square(input_vectors + output_vectors), 1, keep_dims=True))
            self.normalized_embeddings = (input_vectors + output_vectors) / norm
    def train(self, session, num_steps):
        average_loss = 0
        for step in range(num_steps):
            batch_inputs, batch_labels = generate_batch(self.batch_size, self.num_skips, self.skip_window)
            feed_dict = {self.t_inputs: batch_inputs, self.t_labels: batch_labels}
            _, loss_val = session.run([self.train_step, self.loss], feed_dict=feed_dict)
            average_loss += loss_val
            if step % 1000 == 0:  
                if step > 0:
                    average_loss /= 1000
                self.loss_hist.append(average_loss)
                average_loss = 0
    def get_loss_hist(self):
        return self.loss_hist
    def get_weights(self, session):
        return session.run([v for v in tf.trainable_variables() if v.name.startswith(self.name)])
    def set_weights(self, session, weights):
        my_weights = [v for v in tf.trainable_variables() if v.name.startswith(self.name)]
        for i, v in enumerate(my_weights):
            session.run(v.assign(weights[i]))
    def get_name(self):
        return self.name

class FedLearnServer(object):
    def __init__(self, count):
        self.loss_hist = []
        self.server_id = 0
        self.name = format('server_%d' %self.server_id)
        self.learning_rate = 0.4
        self.batch_size = 128
        self.embedding_size = 128
        self.vocabulary_size = 50000
        self.skip_window = 2
        self.num_skips = 1
        self.unigrams = [ c / self.vocabulary_size for token, c in count ]
        self.test_input, self.test_labels= generate_batch(self.batch_size, self.num_skips, self.skip_window)
        self._create_model()
    def _create_model(self):
        with tf.variable_scope(self.name):
            self.t_inputs = tf.placeholder(tf.int32, shape=[self.batch_size],name="t_inputs")
            self.t_labels = tf.placeholder(tf.int32, shape=[self.batch_size, 1],name="t_labels")
            input_ids = self.t_inputs
            labels = tf.reshape(self.t_labels, [self.batch_size])
            labels_matrix = tf.reshape(tf.cast(labels, dtype=tf.int64), [self.batch_size, 1])
            input_vectors = tf.Variable(tf.random_uniform(
                [self.vocabulary_size, self.embedding_size], -1.0, 1.0), name="input_vectors")
            output_vectors = tf.Variable(tf.random_uniform(
                [self.vocabulary_size, self.embedding_size], -1.0, 1.0), name="output_vectors")
            sampled_ids, _, _ = (tf.nn.fixed_unigram_candidate_sampler(
                    true_classes=labels_matrix, num_true=1, num_sampled=200,
                    unique=True, range_max=vocabulary_size, distortion=0.75,
                    unigrams=unigrams))
            center_vects = tf.nn.embedding_lookup(input_vectors, input_ids)
            context_vects = tf.nn.embedding_lookup(output_vectors, labels)
            sampled_vects = tf.nn.embedding_lookup(output_vectors, sampled_ids)
            incorpus_logits = tf.reduce_sum(tf.multiply(center_vects, context_vects), 1)
            incorpus_probabilities = tf.nn.sigmoid(incorpus_logits)
            sampled_logits = tf.matmul(center_vects, sampled_vects, transpose_b=True)
            outcorpus_probabilities = tf.nn.sigmoid(-sampled_logits)
            outcorpus_loss_perexample = tf.reduce_sum(tf.log(outcorpus_probabilities), 1)
            loss_perexample = - tf.log(incorpus_probabilities) - outcorpus_loss_perexample
            self.loss = tf.reduce_sum(loss_perexample) / self.batch_size
            self.train_step = tf.train.GradientDescentOptimizer(.4).minimize(self.loss)
            norm = tf.sqrt(tf.reduce_sum(tf.square(input_vectors + output_vectors), 1, keep_dims=True))
            self.normalized_embeddings = (input_vectors + output_vectors) / norm
    def test(self, session):
        feed_dict = {self.t_inputs: self.test_input, self.t_labels: self.test_labels}
        loss_val = session.run([self.loss], feed_dict=feed_dict)
        self.loss_hist.append(loss_val)
    def get_loss_hist(self):
        return self.loss_hist
    def get_weights(self, session):
        return session.run([v for v in tf.trainable_variables() if v.name.startswith(self.name)])
    def set_weights(self, session, weights):
        my_weights = [v for v in tf.trainable_variables() if v.name.startswith(self.name)]
        for i, v in enumerate(my_weights):
            session.run(v.assign(weights[i]))
    def get_name(self):
        return self.name

class FedLearningSimulator(object):
    def __init__(self, num_clients):
        self.num_clients = num_clients
        self.clients = []
        self.clients_data = []
        self.session = tf.Session()
        for i in range(num_clients):
            self.clients.append(self._create_client(i))
        self.param_server = self._create_server()          
        self.session.run(tf.global_variables_initializer())
    def _create_client(self, i):
        return FedLearnClient( i, count)
    def _create_server(self):
        return FedLearnServer( count)  
    def test(self):
        self.param_server.test(self.session)
        print("Testing complete!\n")   
    def train(self, num_steps, working_list):
        for i, client in enumerate(self.clients):
            if i in working_list:
                print(str(i)+"-th client")
                client.train(self.session, num_steps)
            if i not in working_list:
                client.loss_hist = client.loss_hist + [0]*num_steps
        print("Training Complete")
    def merge_weights(self, working_list):
        client_dataset_weight = []  # client dataset portion
        clients_weights = []        # client trained params
        for i, client in enumerate(self.clients):
            if i in working_list:
                clients_weights.append(client.get_weights(self.session))
                client_dataset_weight.append(1)
        final_weights = self.get_final_weight(clients_weights, client_dataset_weight)
        print("Averaging weights")
        for c in self.clients:
            c.set_weights(self.session, final_weights)
        self.param_server.set_weights(self.session, final_weights)
        print("Set weight. \nParameters updating complete")  
    def get_final_weight(self, clients_weights, client_dataset_weight):
        clients_num = len(client_dataset_weight)
        client_dataset_weight = np.array(client_dataset_weight)
        clients_weights = np.array(clients_weights)
        for i in range(clients_num):
            clients_weights[i] = clients_weights[i]*client_dataset_weight[i]
        final_weights = np.sum(clients_weights, axis = 0)/ sum(client_dataset_weight)
        return final_weights    
    def get_clients_loss_hist(self):
        return [x.get_loss_hist() for x in self.clients]
    def get_server_loss(self):
        return self.param_server.get_loss_hist()

Iteration = 10
K_param = 5
C_param = 1
B_param = 50
E_param = 10000
working_client = max(math.ceil(C_param * K_param) , 1)
simulator = FedLearningSimulator(K_param)

for i in range(Iteration):
    working_list = []
    for j in range(K_param):
        working_list.append(j)
    random.shuffle(working_list)
    working_list = working_list[0:working_client]
    simulator.train(E_param, working_list)
    simulator.merge_weights(working_list)
    simulator.test()
    
width = 16
height = 12
fig= plt.figure(figsize=(width, height))
ax = fig.add_subplot(111)
for i in range(simulator.num_clients):
    client_loss = simulator.clients[i].get_loss_hist()
    plt.plot(client_loss, label=simulator.clients[i].get_name())
server_loss = simulator.param_server.get_loss_hist()
upate_time = [9,19,29,39,49,59,69,79,89,99]
plt.plot(upate_time, server_loss, 'ro', markersize=12, label=simulator.param_server.get_name())
ax.legend()
plt.xlabel('Iteration')
plt.ylabel('Training loss')
fig.show()






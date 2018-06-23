import threading
import multiprocessing
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.signal
from random import randint
from time import sleep
from copy import deepcopy
import os
import itertools
import csv
import time
import random
import argparse
import math
Demand = []

with open("./Demand100000.csv") as csvfile:
    reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in reader:
        Demand.append(int(row[0]))

def new_transition(s, a, demand, LT_s, LT_f, h, b, C_s, C_f, Inv_Max, Inv_Min, cap_fast, cap_slow):
    done = False
    s1 = deepcopy(s)
    reward = 0
    s1[0] += - demand
    s1[LT_f] += a[0]
    s1[LT_s] += a[1]
    reward += math.ceil(a[0]/cap_fast) * C_f +math.ceil(a[1]/cap_slow) * C_s
    if s1[0] >= 0:
        reward += s1[0] * h
    else:
        reward += -s1[0] * b
    s1[0] += s1[1]
    for i in range(1, LT_s):
        s1[i] = s1[i + 1]
    s1[LT_s] = 0
    if (s1[0] > Inv_Max):
        s1[0] = Inv_Max
        done = True
    if s1[0] < Inv_Min:
        s1[0] = Inv_Min
        done = True

    return reward / 1000000, s1, done



# Copies one set of variables to another.
# Used to set worker network parameters to those of global network.
def update_target_graph(from_scope, to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var, to_var in zip(from_vars, to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder


# Discounting function used to calculate discounted returns.
def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


# Used to initialize weights for policy and value output layers
def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)

    return _initializer


def CreateActions(OrderFast, OrderSlow):
    Temp = [0 for z in range((OrderFast + 1) * (OrderSlow + 1))]
    z = 0
    for i in itertools.product(list(range(0, OrderFast + 1)), list(range(0, OrderSlow + 1))):
        Temp[z] = i
        z += 1
    return Temp


class AC_Network():
    def __init__(self, s_size, a_size, scope, trainer,depth_nn_out,activation_nn_hidden,depth_nn_hidden,depth_nn_layers_hidden,activation_nn_out,entropy_factor):
        with tf.variable_scope(scope):
            self.entropy_factor = entropy_factor
            # Input and visual encoding layers
            self.inputs = tf.placeholder(shape=[None, s_size], dtype=tf.float32)

            #structure of the HIDDEN layers network, could be written a bit shorter but haven't done this yet
            if depth_nn_hidden >= 1:
                self.hidden1 = slim.fully_connected(inputs=self.inputs, num_outputs=depth_nn_layers_hidden[0],
                                                    activation_fn=activation_nn_hidden[0])
                self.state_out = slim.fully_connected(inputs=self.hidden1, num_outputs=depth_nn_out,
                                                      activation_fn=activation_nn_out)
            if depth_nn_hidden >= 2:
                self.hidden2 = slim.fully_connected(inputs=self.hidden1, num_outputs=depth_nn_layers_hidden[1],
                                                    activation_fn=activation_nn_hidden[1])
                self.state_out = slim.fully_connected(inputs=self.hidden2, num_outputs=depth_nn_out,
                                                      activation_fn=activation_nn_out)
            if depth_nn_hidden >= 3:
                self.hidden3 = slim.fully_connected(inputs=self.hidden2, num_outputs=depth_nn_layers_hidden[2],
                                                    activation_fn=activation_nn_hidden[2])
                self.state_out = slim.fully_connected(inputs=self.hidden3, num_outputs=depth_nn_out,
                                                      activation_fn=activation_nn_out)
            if depth_nn_hidden >= 4:
                self.hidden4 = slim.fully_connected(inputs=self.hidden3, num_outputs=depth_nn_layers_hidden[3],
                                                    activation_fn=activation_nn_hidden[3])
                self.state_out = slim.fully_connected(inputs=self.hidden4, num_outputs=depth_nn_out,
                                                      activation_fn=activation_nn_out)

            # This is the ACTOR output
            self.policy = slim.fully_connected(self.state_out, a_size,
                                               activation_fn=tf.nn.softmax,
                                               weights_initializer=normalized_columns_initializer(0.01),
                                               biases_initializer=None)


            #this is the CRITIC ouput
            self.value = slim.fully_connected(self.state_out, 1,
                                              activation_fn=None,
                                              weights_initializer=normalized_columns_initializer(1.0),
                                              biases_initializer=None)

            # Only the worker network need ops for loss functions and gradient updating.
            if scope != 'global':
                self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
                self.actions_onehot = tf.one_hot(self.actions, a_size, dtype=tf.float32)
                self.target_v = tf.placeholder(shape=[None], dtype=tf.float32)
                self.advantages = tf.placeholder(shape=[None], dtype=tf.float32)

                self.responsible_outputs = tf.reduce_sum(self.policy * self.actions_onehot, [1])

                # Loss functions
                self.value_loss = 0.5 * tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.value, [-1])))
                self.entropy =  -tf.reduce_sum(self.policy * tf.log(self.policy + 1e-10))
                self.policy_loss = tf.reduce_sum(tf.log(self.responsible_outputs) * self.advantages)
                self.loss = 0.5 * self.value_loss + self.policy_loss - self.entropy * self.entropy_factor

                # Get gradients from local network using local losses
                local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                self.gradients = tf.gradients(self.loss, local_vars)
                self.var_norms = tf.global_norm(local_vars)
                grads, self.grad_norms = tf.clip_by_global_norm(self.gradients, 40)

                # Apply local gradients to global network
                global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                self.apply_grads = trainer.apply_gradients(zip(grads, global_vars))


class Worker():
    def __init__(self, name, s_size, a_size, trainer, model_path, best_path,log_path, global_episodes,depth_nn_out,
                 activation_nn_hidden,depth_nn_hidden,depth_nn_layers_hidden,activation_nn_out,entropy_factor):
        self.name = "worker_" + str(name)
        self.number = name
        self.model_path = model_path
        self.best_path = best_path
        self.log_path = log_path
        self.trainer = trainer
        self.global_episodes = global_episodes
        self.increment = self.global_episodes.assign_add(1)
        self.no_improvement = 0#tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
       # self.no_improvement_increment = self.no_improvement.assign_add(1)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_mean_values = []
        self.summary_writer = tf.summary.FileWriter(
            model_path + str(self.number) + str(time.strftime(" %Y%m%d-%H%M%S")))
        # Create the local copy of the network and the tensorflow op to copy global parameters to local network
        self.local_AC = AC_Network(s_size, a_size, self.name, trainer,depth_nn_out,activation_nn_hidden,depth_nn_hidden,
                                   depth_nn_layers_hidden,activation_nn_out,entropy_factor)
        self.update_local_ops = update_target_graph('global', self.name)
        self.actions = np.identity(a_size, dtype=bool).tolist()
        self.bool_evaluating = None
        self.best_median_solution = 9999999999999
        self.best_mean_solution = 9999999999999
        self.mean_solution_vector = []
        self.median_solution_vector = []

    def train(self, rollout, sess, gamma, bootstrap_value):
        rollout = np.array(rollout)
        observations = rollout[:, 0]
        actions = rollout[:, 1]
        rewards = rollout[:, 2]
        #next_observations = rollout[:, 3]
        values = rollout[:, 5]

        # Here we take the rewards and values from the rollout, and use them to
        # generate the advantage and discounted returns.
        # The advantage function uses "Generalized Advantage Estimation"
        self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = discount(self.rewards_plus, gamma)[:-1]
        self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
        advantages = rewards + gamma * self.value_plus[1:] - self.value_plus[:-1]
        advantages = discount(advantages, gamma)

        # Update the global network using gradients from loss
        # Generate network statistics to periodically save
        feed_dict = {self.local_AC.target_v: discounted_rewards,
                     self.local_AC.inputs: np.vstack(observations),
                     self.local_AC.actions: actions,
                     self.local_AC.advantages: advantages}

        v_l, p_l, e_l, g_n, v_n, Policy, _ = sess.run(
            [self.local_AC.value_loss,
             self.local_AC.policy_loss,
             self.local_AC.entropy,
             self.local_AC.grad_norms,
             self.local_AC.var_norms,
             self.local_AC.policy,
             self.local_AC.apply_grads],
            feed_dict=feed_dict)
        return v_l / len(rollout), p_l / len(rollout), e_l / len(rollout), g_n, v_n

    def work(self, max_episode_length, gamma, sess, coord, saver, saver_best,Demand, LT_s, LT_f, h, b, C_s, C_f, InvMax,
             InvMin,cap_fast, cap_slow,initial_state,Penalty,Demand_Max,max_training_episodes,actions,
             p_len_episode_buffer,max_no_improvement,pick_largest,verbose,entropy_decay,entropy_min,cut_10):
        episode_count = sess.run(self.global_episodes)
        try:
            with open(self.log_path+'best_median_solution.csv', newline='') as csvfile:
                csvreader = csv.reader(csvfile,delimiter=';', quotechar='|')
                best_median_vector = []
                for row in csvreader:
                    best_median_vector.append(float(row[0]))
                best_median = np.min(best_median_vector)
                #print(best_median)
        except:
            best_median = 999999999999

        print('Best median found:', best_median)

        print("Starting worker " + str(self.number))
        with sess.as_default(), sess.graph.as_default():
            #Stop training after max_training_episodes (2nd part is an early stopping criterion when I am not converging
            while episode_count < max_training_episodes and (episode_count < cut_10 or self.best_median_solution < 1.1*best_median):

                #Here I evaluate every 50 episodes on the same sample path (this is for a part of my paper but is very
                #time intensive ==> put this to a very high number such that you don't lose time here
                if (episode_count % 50 == 0):
                    self.bool_evaluating = True
                else:
                    self.bool_evaluating = None
                #Code to reduce entropy factor while training:
                #if (episode_count % 10 == 0 and self.local_AC.entropy_factor > entropy_min):
                #    self.local_AC.entropy_factor *= entropy_decay
                #    print('CHECK',self.local_AC.entropy_factor)
                sess.run(self.update_local_ops)
                eval_performance = []
                for i in range(10):
                    episode_buffer = []
                    episode_values = []

                    eval_buffer = []
                    episode_reward = 0
                    episode_step_count = 0
                    d = False

                    if self.bool_evaluating == True:
                        self.inv_vect = np.array(initial_state)
                    else:
                        self.inv_vect = np.array(
                            initial_state)
                    s = deepcopy(self.inv_vect)

                    self.no_improvement+=1
                    while (d == False and episode_step_count < max_episode_length - 1):

                            # Take an action using probabilities from policy network output.
                            a_dist, v = sess.run([self.local_AC.policy, self.local_AC.value],
                                                 feed_dict={self.local_AC.inputs: [s]})  # ,

                            #This is to test what would happen if you pick the best action instead of sample from actor
                            # distribution. Mainly yo compare after convergence between deterministic and stochastic
                            # policy:
                            if  pick_largest:# or self.bool_evaluating:
                                a = np.argmax(a_dist[0])
                            else:
                                a = np.random.choice(np.arange(len(a_dist[0])), p=a_dist[0])

                            if self.bool_evaluating == True:
                                r,s1,d = new_transition(s, actions[a], Demand[episode_step_count*(i+1)],
                                                        LT_s, LT_f, h, b, C_s, C_f, InvMax, InvMin, cap_fast, cap_slow)
                                d = False

                            else:
                                r, s1, d = new_transition(s, actions[a], random.randint(0, Demand_Max), LT_s, LT_f, h, b, C_s,
                                                          C_f, InvMax, InvMin, cap_fast, cap_slow)
                                d = False

                            if self.bool_evaluating == True:
                                eval_buffer.append([s, actions[a], r, s1, d, v[0, 0]])
                            episode_buffer.append([s, a, r, s1, d, v[0, 0]])

                            episode_values.append(v[0, 0])
                            episode_reward += r
                            s = deepcopy(s1)
                            episode_step_count += 1

                            # If the episode hasn't ended, but the experience buffer is full, then we
                            # make an update step using that experience rollout.
                            if len(episode_buffer) == p_len_episode_buffer and d != True and episode_step_count != max_episode_length - 1 and self.bool_evaluating != True:
                                # Since we don't know what the true final return is, we "bootstrap" from our current
                                # value estimation.
                                v1 = sess.run(self.local_AC.value,
                                              feed_dict={self.local_AC.inputs: [s]})[0, 0]
                                v_l, p_l, e_l, g_n, v_n = self.train(episode_buffer, sess, gamma, v1)

                                #print(v_l,p_l,e_l,g_n,v_n)
                                #print("v_l", v_l, v_l / (v_l - p_l - self.local_AC.entropy_factor * e_l), "p_l", p_l,
                                #      -p_l / (v_l - p_l - self.local_AC.entropy_factor * e_l), "e_l",
                                #      self.local_AC.entropy_factor * e_l,
                                #      self.local_AC.entropy_factor * (-e_l) / (v_l - p_l - self.local_AC.entropy_factor * e_l),
                                #      "g_n", g_n, "v_n", v_n)
                                episode_buffer = []
                                sess.run(self.update_local_ops)

                    if(self.bool_evaluating != True):
                        break
                        #again, for evaluating purposes on the same sample path I need to do this, you can ignore this
                    else:
                        eval_performance.append(episode_reward/episode_step_count)
                        #print(i,eval_performance)
                if(self.bool_evaluating):
                    #print('PRIOR',eval_performance)
                    mean_performance = np.mean(eval_performance)
                    std_performance = np.std(eval_performance)
                    median_performance = np.median(eval_performance)

                #These are just some outputs I need for my analysis
                if self.bool_evaluating == True:
                    #if(verbose): print("EVALUATION", episode_reward / episode_step_count, episode_step_count)
                    if (verbose): print("EVALUATION",median_performance, self.best_median_solution,self.no_improvement,episode_count)
                    if (median_performance < self.best_median_solution):# and episode_step_count == max_episode_length - 1):
                        self.best_median_solution = median_performance#episode_reward / episode_step_count
                        self.median_solution_vector = eval_performance

                        f= open(self.best_path +"/best_median_solution%i.txt"%self.number,"w")
                        f.write(str(self.best_median_solution) + ' ' + str(std_performance) + ' ' +  str(self.median_solution_vector) +' '+str (episode_step_count))
                        f.close()
                        saver_best.save(sess, self.best_path + '/Train_' + str(self.number) + '/model_median_' + ' ' + str(
                            episode_count) + '.cptk')
                        self.no_improvement=0#sess.run(self.no_improvement.assign(0))
                    if (
                        mean_performance < self.best_mean_solution):  # and episode_step_count == max_episode_length - 1):
                        self.best_mean_solution = mean_performance  # episode_reward / episode_step_count
                        self.mean_solution_vector = eval_performance

                        f = open(self.best_path + "/best_mean_solution%i.txt" % self.number, "w")
                        f.write(str(self.best_mean_solution) + ' ' + str(std_performance) + ' ' + str(self.mean_solution_vector) + ' ' + str(
                            episode_step_count))
                        f.close()
                        saver_best.save(sess, self.best_path + '/Train_' + str(
                            self.number) + '/model_mean_' + ' ' + str(
                            episode_count) + '.cptk')
                        # print(sess.run(self.number,self.no_improvement))

                if self.bool_evaluating != True:
                    self.episode_rewards.append(episode_reward)
                    self.episode_lengths.append(episode_step_count)
                    self.episode_mean_values.append(np.mean(episode_values))


                #save model every 250 periods
                if episode_count % 250 == 0 and self.name == 'worker_0':
                    saver.save(sess, self.model_path + '/model-' + str(episode_count) + '.cptk')

                if self.bool_evaluating != True:
                    mean_reward = np.mean(self.episode_rewards[-5:])
                    mean_length = np.mean(self.episode_lengths[-5:])
                    mean_value = np.mean(self.episode_mean_values[-5:])

                    #this is for nice visualization in Tensorboard (very useful for checking during training)
                    summary = tf.Summary()
                    summary.value.add(tag='Perf/Reward per period', simple_value=float(mean_reward / mean_length))
                    summary.value.add(tag='Perf/Length', simple_value=float(mean_length))
                    summary.value.add(tag='Perf/Value', simple_value=float(mean_value))
                    summary.value.add(tag='Losses/Value Loss', simple_value=float(v_l))
                    summary.value.add(tag='Losses/Policy Loss', simple_value=float(p_l))
                    summary.value.add(tag='Losses/Entropy', simple_value=float(e_l))
                    summary.value.add(tag='Losses/Grad Norm', simple_value=float(g_n))
                    summary.value.add(tag='Losses/Var Norm', simple_value=float(v_n))

                    self.summary_writer.add_summary(summary, episode_count)
                    #tf.summary.FileWriter(model_path, sess.graph)
                    self.summary_writer.flush()
                if self.name == 'worker_0':
                    sess.run(self.increment)


                episode_count += 1
                if self.no_improvement >= max_no_improvement:
                    break


def NewCreateStates(LT_f,LT_s,Inv_Max,Inv_Min,O_f,O_s):
    Temp = []
    total_pipe = []
    total_pipe.append(range(Inv_Min,Inv_Max+1))
    for _ in range(1,LT_f+1):
        total_pipe.append(range(O_f+O_s+1))
    for _ in range(LT_f+1,LT_s):
        total_pipe.append(range(O_s+1))
    for index,i in enumerate(itertools.product(*total_pipe)):
        Temp.append(list(i))
        Temp[index].append(0)
    return Temp

def write_parameters(model_path, depth_nn_hidden, depth_nn_layers_hidden, depth_nn_out, entropy_factor,
                     activation_nn_hidden, activation_nn_out, learning_rate, optimizer, activations,
                     p_len_episode_buffer, max_episode_length, OrderFast, OrderSlow, LT_s, LT_f, InvMax,
                     max_training_episodes,h,b,C_f,C_s,InvMin,Penalty,initial_state,nb_workers,cap_fast,cap_slow,cut_10):
    #Sorry, bad readability here, just writing away the parameters of my model, have not put attention into this
    f = open(model_path + "/Parameters.txt", "w")
    parameters = {}
    f.write("depth_nn_hidden: " + str(depth_nn_hidden))
    parameters["depth_nn_hidden"] = depth_nn_hidden
    f.write("\ndepth_nn_layers_hidden " + str(depth_nn_layers_hidden))
    parameters["depth_nn_layers_hidden"] = depth_nn_layers_hidden
    f.write("\ndepth_nn_out: " + str(depth_nn_out))
    parameters["depth_nn_out"] = depth_nn_out
    f.write("\nentropy_factor " + str(entropy_factor))
    parameters["entropy_factor"] = entropy_factor
    f.write("\nactivation_nn_hidden: " + str(activation_nn_hidden))
    parameters["activation_nn_hidden"] = activation_nn_hidden
    f.write("\nactivation_nn_out " + str(activation_nn_out))
    parameters["activation_nn_out"] = activation_nn_out
    f.write("\nLearning Rate: " + str(learning_rate))
    parameters["learning_rate"] = learning_rate
    f.write("\noptimizer " + str(optimizer))
    parameters["optimizer"] = optimizer
    f.write("\nactivations: " + str(activations))
    parameters["activations"] = activations
    f.write("\np_len_episode_buffer " + str(p_len_episode_buffer))
    parameters["p_len_episode_buffer"] = p_len_episode_buffer
    f.write("\nmax_episode_length: " + str(max_episode_length))
    parameters["max_episode_length"] = max_episode_length
    f.write("\nOrderFast " + str(OrderFast))
    parameters["OrderFast"] = OrderFast
    f.write("\nOrderSlow " + str(OrderSlow))
    parameters["OrderSlow"] = OrderSlow
    f.write("\nLT_s " + str(LT_s))
    parameters["LT_s"] = LT_s
    f.write("\nLT_f " + str(LT_f))
    parameters["LT_f"] = LT_f
    f.write("\nh " + str(h))
    parameters["h"] = h
    f.write("\nb " + str(b))
    parameters["b"] = b
    f.write("\nC_f " + str(C_f))
    parameters["C_f"] = C_f
    f.write("\nC_s " + str(C_s))
    parameters["C_s"] = C_s
    f.write("\nInvMin " + str(InvMin))
    parameters["InvMin"] = InvMin
    f.write("\nInvMax " + str(InvMax))
    parameters["InvMax"] = InvMax
    f.write("\nPenalty " + str(Penalty))
    parameters["Penalty"] = Penalty
    f.write("\ninitial_state " + str(initial_state))
    parameters["initial_state"] = initial_state
    f.write("\nmax_training_episodes " + str(max_training_episodes))
    parameters["max_training_episodes"] = max_training_episodes
    f.write("\nnb_workers" + str(nb_workers))
    parameters["nb_workers"] = nb_workers
    f.write("\ncap_fast" + str(cap_fast))
    parameters["cap_fast"] = cap_fast
    f.write("\ncap_slow" + str(cap_slow))
    parameters["cap_slow"] = cap_slow
    f.write("\ncut_10" + str(cut_10))
    parameters['cut_10'] = cut_10
    f.close()
    return parameters


def objective(parameters):
    # Again bad readability here, just setting local variables
    Demand_Max = parameters['Demand_Max']
    OrderFast = parameters['OrderFast']
    OrderSlow = parameters['OrderSlow']
    Penalty = parameters['penalty']
    LT_f = parameters['LT_f']
    LT_s = parameters['LT_s']
    h = parameters['h']
    b = parameters['b']
    C_f = parameters['C_f']
    C_s = parameters['C_s']
    cap_fast = parameters['cap_fast']
    cap_slow = parameters['cap_slow']
    max_training_episodes = parameters['max_training_episodes']
    learning_rate = parameters['initial_lr']
    entropy_factor = parameters['entropy']
    gamma = parameters['gamma']
    max_no_improvement = parameters['max_no_improvement']
    max_training_episodes = parameters['max_training_episodes']
    depth_nn_hidden = parameters['depth_nn_hidden']
    depth_nn_layers_hidden = parameters['depth_nn_layers_hidden']
    depth_nn_out = parameters['depth_nn_out']
    p_len_episode_buffer = parameters['p_len_episode_buffer']
    initial_state = parameters['initial_state']
    initial_state=initial_state*LT_s
    initial_state.append(0)
    InvMax = parameters['invmax']
    InvMin = parameters['invmin']
    training = parameters['training']
    pick_largest = parameters['high']
    nb_workers = parameters['nbworkers']
    verbose = parameters['verbose']
    entropy_decay = parameters['entropy_decay']
    entropy_min = parameters['entropy_min']
    cut_10 = parameters['cut_10']
    activation_nn_hidden = [tf.nn.relu, tf.nn.relu, tf.nn.relu, tf.nn.relu]
    activation_nn_out = tf.nn.relu
    optimizer = tf.train.AdamOptimizer(learning_rate)
    activations = [tf.nn.relu, tf.nn.relu]
    max_episode_length = parameters['max_episode_length']



    #Creating Actions sizes
    actions = CreateActions(OrderFast, OrderSlow)
    a_size = len(actions)  # all possible actions
    s_size = LT_s + 1 #size of 1 state (inventory vector)

    tf.reset_default_graph()
    if training:
        load_model = False
    else:
        load_model=True

    model_path = 'Logs/Logs_' + str(time.strftime("%Y%m%d-%H%M%S")) + '/model'
    best_path = 'Logs/Logs_' + str(time.strftime("%Y%m%d-%H%M%S")) + '/best'
    log_path = 'Logs/'

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    global_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)

    trainer = optimizer
    master_network = AC_Network(s_size, a_size, 'global', None,depth_nn_out,activation_nn_hidden,depth_nn_hidden,
                                depth_nn_layers_hidden,activation_nn_out,entropy_factor)  # Generate global network
    num_workers = nb_workers
    workers = []

    parameters = write_parameters(model_path, depth_nn_hidden, depth_nn_layers_hidden, depth_nn_out, entropy_factor,
                     activation_nn_hidden, activation_nn_out, learning_rate, optimizer, activations,
                     p_len_episode_buffer, max_episode_length, OrderFast, OrderSlow, LT_s, LT_f, InvMax,
                     max_training_episodes,h,b,C_f,C_s,InvMin,Penalty,initial_state,nb_workers,cap_fast,cap_slow,cut_10)

    # Create worker classes
    for i in range(num_workers):
        if not os.path.exists(best_path + '/Train_' + str(i)):
            os.makedirs(best_path + '/Train_' + str(i))
        workers.append(Worker(i, s_size, a_size, trainer, model_path, best_path,log_path, global_episodes,depth_nn_out,
                              activation_nn_hidden,depth_nn_hidden,depth_nn_layers_hidden,activation_nn_out,entropy_factor))
    saver = tf.train.Saver(max_to_keep=5)
    saver_best = tf.train.Saver(max_to_keep=None)

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        if load_model == True:

            print('Loading Model...')
            ckpt = tf.train.get_checkpoint_state('./')
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())

        # This is where the asynchronous magic happens.
        # Start the "work" process for each worker in a separate threat.

        if(training):
            worker_threads = []
            temp_best_mean_solutions = np.zeros(len(workers))
            temp_best_median_solutions = np.zeros(len(workers))
            for worker in workers:
                worker_work = lambda: worker.work(max_episode_length, gamma, sess, coord, saver, saver_best,Demand, LT_s,
                                                  LT_f, h, b, C_s, C_f,InvMax, InvMin, cap_fast, cap_slow,initial_state,
                                                  Penalty,Demand_Max,max_training_episodes,actions,p_len_episode_buffer,
                                                  max_no_improvement,pick_largest,verbose,entropy_decay,entropy_min, cut_10)
                t = threading.Thread(target=(worker_work))
                t.start()
                sleep(0.5)
                worker_threads.append(t)
            coord.join(worker_threads)
            for index, worker in enumerate(workers):
                temp_best_mean_solutions[index] = worker.best_mean_solution
                temp_best_median_solutions[index] = worker.best_median_solution

            best_mean_solution_found = np.min(temp_best_mean_solutions)
            best_median_solution_found = np.min(temp_best_median_solutions)



            #Writing down results for my analysis
            with open(log_path+'best_mean_solution.csv','a') as f:
                f.write(str(best_mean_solution_found)+';'+str(best_path)+';')
                for key, value in parameters.items():
                    f.write(str(value) + ';')
                for key,value in parameters.items():
                    f.write(str(key)+';')
                for item in worker.median_solution_vector:
                    f.write(str(item) + ';')
                f.write('\n')

            # Writing down results for my analysis
            with open(log_path+'best_median_solution.csv','a') as f:
                f.write(str(best_median_solution_found)+';'+str(best_path)+';')
                for key, value in parameters.items():
                    f.write(str(value) + ';')
                for key,value in parameters.items():
                    f.write(str(key)+';')
                for item in worker.mean_solution_vector:
                    f.write(str(item) + ';')
                f.write('\n')


        # You can ignotre this part, I just do some quick tests here on different trained models
        else:
            States = NewCreateStates(LT_f,LT_s,10,-10,OrderFast,OrderSlow)
            print(States)
            policy_fast = []
            policy_slow = []
            A3C_policy = []
            for index, state in enumerate(States):
                prob_vector = sess.run(workers[0].local_AC.policy,feed_dict={workers[0].local_AC.inputs:[state]})[0]
                A3C_policy.append(prob_vector)

                action_prob_fast = np.zeros(OrderFast + 1)
                action_prob_slow = np.zeros(OrderSlow + 1)

                for i in range(len(actions)):
                    action_prob_fast[actions[i][0]] += prob_vector[i]
                    action_prob_slow[actions[i][1]] += prob_vector[i]
                print(state,np.argmax(action_prob_fast),np.argmax(action_prob_slow),"FAST", action_prob_fast,"SLOW", action_prob_slow)
                policy_fast.append(deepcopy(action_prob_fast))
                policy_slow.append(deepcopy(action_prob_slow))

            np.savetxt('A3C_policy.csv',A3C_policy,delimiter=';')
            with open('cost.csv', 'w') as f:
               for index, i in enumerate(States):
                   for j in States[index]:
                        f.write(str(j) + ';')
                   f.write(';')
                   for j in policy_fast[index]:
                        f.write(str(j) + ';')
                   f.write(';')
                   for j in policy_slow[index]:
                        f.write(str(j) + ';')
                   f.write(';')
                   f.write('\n')
        return best_median_solution_found


#this is for using Bayesion optimization, implementation bit quick and dirty
def obj_bo(list):
    parameters = {}
    parameters['initial_lr'] = list[0]
    parameters['entropy'] = list[1]

    if (list[2]==0):
        parameters['depth_nn_hidden'] = 1
        parameters["depth_nn_layers_hidden"] = [8,0,0,0]
        parameters["depth_nn_out"] = 4
    if (list[2]==1):
        parameters["depth_nn_hidden"] = 2
        parameters["depth_nn_layers_hidden"] = [40,20,0,0]
        parameters["depth_nn_out"] = 10
    if (list[2] == 2):
        parameters["depth_nn_hidden"] = 3
        parameters["depth_nn_layers_hidden"] = [150, 120, 80, 0]
        parameters["depth_nn_out"] = 20

    if (list[2] == 3):
        parameters["depth_nn_hidden"] = 4
        parameters["depth_nn_layers_hidden"] = [200, 180, 100, 70]
        parameters["depth_nn_out"] = 40

    parameters['p_len_episode_buffer'] = list[3]
    parameters['cut_10'] = list[4]
    parameters['max_no_improvement'] = list[5]
    parameters['LT_s'] = list[6]
    parameters['OrderFast'] = list[7]
    parameters['OrderSlow'] = list[8]
    parameters['cap_slow'] = list[9]
    parameters['cap_fast'] = list[10]
    parameters['C_f'] = list[11]
    parameters['Demand_Max'] = 4
    parameters['b'] = list[12]

    parameters['penalty'] = 1
    parameters['LT_f'] = 0
    parameters['h'] = 5
    parameters['C_s'] = 100
    parameters['gamma'] = 0.99
    parameters['max_training_episodes'] = 10000000
    parameters['invmax'] = 40  # (LT_s+1)*(2*Demand_Max+1)
    parameters['invmin'] = -40 # -(LT_s+1)*(2*Demand_Max)
    parameters['training'] = True
    parameters['high'] = False
    parameters['nbworkers'] = 4
    parameters['verbose'] = True
    parameters['entropy_decay'] = 1
    parameters['entropy_min'] = 1
    parameters['initial_state'] = [3]
    parameters['max_episode_length'] = 1000
    result = objective(parameters)

    return result



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-lr', '--initial_lr', default=0.0001, type=float,
                        help="Initial value for the learning rate.  Default = 0.001",
                        dest="initial_lr")
    parser.add_argument('--entropy', default=0.000001, type=float,
                        help="Strength of the entropy regularization term (needed for actor-critic). Default = 0.01",
                        dest="entropy")
    parser.add_argument('--gamma', default=0.99, type=float, help="Discount factor. Default = 0.99", dest="gamma")
    parser.add_argument('--max_no_improvement', default=2500, type=float, help="max_no_improvement. Default = 5000", dest="max_no_improvement")
    parser.add_argument('--max_training_episodes', default=10000000, type=float, help="max_training_episodes. Default = 10000000",
                        dest="max_training_episodes")
    parser.add_argument('--depth_nn_hidden', default=4, type=float,
                        help="depth_nn_hidden. Default = 3",
                        dest="depth_nn_hidden")
    parser.add_argument('--depth_nn_out', default=20, type=float,
                        help="depth_nn_out. Default = 20",
                        dest="depth_nn_out")
    parser.add_argument('--depth_nn_layers_hidden', default=[150,120,80,40], type=list,
                        help="depth_nn_layers_hidden. Default = [40,20,10,10]",
                        dest="depth_nn_layers_hidden")
    parser.add_argument('--p_len_episode_buffer', default=20, type=float,
                        help="p_len_episode_buffer. Default = 20",
                        dest="p_len_episode_buffer")
    parser.add_argument('--initial_state', default=[3], type=float,
                        help="initial_state. Default = [3,0]",
                        dest="initial_state")
    parser.add_argument('--invmax', default=40, type=float,
                        help="invmax. Default = 150",
                        dest="invmax")
    parser.add_argument('--invmin', default=-40 , type=float,
                        help="invmin. Default = -15",
                        dest="invmin")
    parser.add_argument('--training', default= True, type=float,
                        help="training. Default = True",
                        dest="training")
    parser.add_argument('--high', default= False, type=float,
                        help="Pick largest likelihood. Default = False",
                        dest="high")
    parser.add_argument('--nbworkers', default= 4, type=float,
                        help="Number of A3C workers. Default = 4",
                        dest="nbworkers")
    parser.add_argument('--verbose', default= True, type=str,
                        help="Print evaluation results. Default = False",
                        dest="verbose")
    parser.add_argument('--entropy_decay', default= 0.9, type=float,
                        help="Entropy_decay. Default = 0.95",
                        dest="entropy_decay")
    parser.add_argument('--entropy_min', default= 1, type=float,
                        help="entropy_min. Default = 0",
                        dest="entropy_min")
    parser.add_argument('-Demand_Max', '--Demand_Max', default=4, type=float,
                        help="Demand_Max. Default = 4",
                        dest="Demand_Max")
    parser.add_argument('--OrderFast', default=5, type=int,
                        help="OrderFast. Default = 5",
                        dest="OrderFast")
    parser.add_argument('--OrderSlow', default=5, type=int, help="OrderSlow. Default = 5", dest="OrderSlow")
    parser.add_argument('--LT_s', default=1, type=int, help="LT_s. Default = 1", dest="LT_s")
    parser.add_argument('--LT_f', default=0, type=int, help="LT_f. Default = 0",
                        dest="LT_f")
    parser.add_argument('--cap_slow', default=1, type=float,
                        help="cap_slow. Default = 1",
                        dest="cap_slow")
    parser.add_argument('--cap_fast', default=1, type=float,
                        help="cap_fast. Default = 1",
                        dest="cap_fast")
    parser.add_argument('--C_s', default=100, type=float,
                        help="C_s. Default = 100",
                        dest="C_s")
    parser.add_argument('--C_f', default=150, type=float,
                        help="C_f. Default = 150",
                        dest="C_f")
    parser.add_argument('--h', default=5, type=float,
                        help="h. Default = 5",
                        dest="h")
    parser.add_argument('--b', default=495, type=str,
                        help="b. Default = 495",
                        dest="b")
    parser.add_argument('--penalty', default=1, type=str,
                        help="penalty. Default = 1",
                        dest="penalty")
    parser.add_argument('--max_time', default=120, type=str,
                        help="max_time. Default = 120",
                        dest="max_time")
    parser.add_argument('--max_episode_length', default=1000, type=str,
                        help="max_episode_length. Default = 100",
                        dest="max_episode_length")
    parser.add_argument('--cut_10', default=2000, type=float,
                        help="cut_10. Default = 2000",
                        dest="cut_10")
    args = parser.parse_args()
    parameters = vars(args)
    objective(parameters)
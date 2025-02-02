#########################################################################
#           Self Supervised Dynamics Model Implementation
#   Author: Utkarsh Mishra (umishra@me.iitr.ac.in)
#   release: Debugging, (Not Completed)


import tensorflow as tf
import numpy as np


class Dynamics:

    def __init__(self, sess, state_vector_size, action_vector_size, latent_space_size, learning_rate):
        self._sess = sess

        self._state = tf.placeholder(dtype=tf.float32, shape=state_vector_size, name='state')
        self._next_state = tf.placeholder(dtype=tf.float32, shape=state_vector_size, name='nstate')
        self._action = tf.placeholder(dtype=tf.float32, shape=(1,action_vector_size), name='action')

        self.latent_space_size = latent_space_size
        self.state_vector_size = state_vector_size
        self.action_vector_size = action_vector_size

        ###################### Latent Feature Extraction Layer ###################################

        self.latent_layer_1 = tf.layers.dense(inputs=tf.expand_dims(self._state, 0), units=64, activation=tf.nn.relu, kernel_initializer=tf.zeros_initializer())
        self.latent_layer_2 = tf.layers.dense(inputs=tf.expand_dims(self.latent_layer_1, 0), units=64, activation=tf.nn.relu, kernel_initializer=tf.zeros_initializer())
        self.latent_layer_out = tf.layers.flatten(inputs=tf.layers.dense(inputs=self.latent_layer_2, units=self.latent_space_size, kernel_initializer=tf.zeros_initializer()))

        # self._latent_state = self.latent_block(self._state)
        # self._latent_nstate = self.latent_block(self._next_state)

        self.latent_layer_1n = tf.layers.dense(inputs=tf.expand_dims(self._next_state, 0), units=64, activation=tf.nn.relu, kernel_initializer=tf.zeros_initializer())
        self.latent_layer_2n = tf.layers.dense(inputs=tf.expand_dims(self.latent_layer_1n, 0), units=64, activation=tf.nn.relu, kernel_initializer=tf.zeros_initializer())
        self.latent_layer_outn = tf.layers.flatten(inputs=tf.layers.dense(inputs=self.latent_layer_2n, units=self.latent_space_size, kernel_initializer=tf.zeros_initializer()))


        # self._merged_latent_states = tf.concat([self._latent_state, self._latent_nstate], 1)
        self._merged_latent_states = tf.concat([self.latent_layer_out, self.latent_layer_outn], 1)


        ##########################################################################################

        ################################ Inverse Dynamics Block ##################################

        self.inverse_dynamics_1 = tf.layers.dense(inputs=tf.expand_dims(self._merged_latent_states, 0), units=128, activation=tf.nn.relu, kernel_initializer=tf.zeros_initializer())
        self.inverse_dynamics_2 = tf.layers.dense(inputs=tf.expand_dims(self.inverse_dynamics_1, 0), units=128, activation=tf.nn.relu, kernel_initializer=tf.zeros_initializer())
        self.inverse_dynamics_action = tf.layers.flatten(inputs=tf.layers.dense(inputs=self.inverse_dynamics_2, units=self.action_vector_size, kernel_initializer=tf.zeros_initializer()))

        ##########################################################################################

        ################################# Latent + Inverse Dynamics Loss #########################

        self._loss_latent = tf.reduce_mean(tf.square(self.inverse_dynamics_action - self._action))

        self._optimizer_latent = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self._train_op_latent = self._optimizer_latent.minimize(self._loss_latent)

        #########################################################################################

        ################################# Forward Dynamics Model ################################

        self._merged_lstate_action = tf.concat([self.latent_layer_out, self._action], 1)
        self.forward_dynamics_1 = tf.layers.dense(inputs=tf.expand_dims(self._merged_lstate_action, 0), units=128, activation=tf.nn.relu, kernel_initializer=tf.zeros_initializer())
        self.forward_dynamics_2 = tf.layers.dense(inputs=tf.expand_dims(self.forward_dynamics_1, 0), units=128, activation=tf.nn.relu, kernel_initializer=tf.zeros_initializer())
        self.forward_dynamics_lstate = tf.layers.flatten(inputs=tf.layers.dense(inputs=self.forward_dynamics_2, units=self.latent_space_size, kernel_initializer=tf.zeros_initializer()))

        #########################################################################################

        ################################# Forward Dynamics Loss #################################

        self._loss_forward = tf.reduce_mean(tf.square(self.forward_dynamics_lstate - self.latent_layer_outn))

        self._optimizer_forward = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self._train_op_forward = self._optimizer_forward.minimize(self._loss_forward)

        #########################################################################################

        ######################## Inverse Latent Feature Layer ###################################

        self._inv_latent_nstate = self.inv_latent_block(self.forward_dynamics_lstate)

        self._loss_invlatent = tf.reduce_mean(tf.square(self._inv_latent_nstate - self._next_state))

        self._optimizer_invlatent = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self._train_op_invlatent = self._optimizer_invlatent.minimize(self._loss_invlatent)

        ##########################################################################################

        self._loss = self._loss_forward + self._loss_latent + self._loss_invlatent
        self._optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self._train_op = self._optimizer.minimize(self._loss)


    def latent_block(self, state_tensor):

        latent_layer_1 = tf.layers.dense(inputs=tf.expand_dims(state_tensor, 0), units=64, activation=tf.nn.relu, kernel_initializer=tf.zeros_initializer())
        latent_layer_2 = tf.layers.dense(inputs=tf.expand_dims(latent_layer_1, 0), units=64, activation=tf.nn.relu, kernel_initializer=tf.zeros_initializer())
        latent_layer_out = tf.layers.dense(inputs=latent_layer_2, units=self.latent_space_size, kernel_initializer=tf.zeros_initializer())

        return tf.layers.flatten(inputs=latent_layer_out) 

    def inv_latent_block(self, latent_state_tensor):

        inv_latent_layer_1 = tf.layers.dense(inputs=tf.expand_dims(latent_state_tensor, 0), units=64, activation=tf.nn.relu, kernel_initializer=tf.zeros_initializer())
        inv_latent_layer_2 = tf.layers.dense(inputs=tf.expand_dims(inv_latent_layer_1, 0), units=64, activation=tf.nn.relu, kernel_initializer=tf.zeros_initializer())
        inv_latent_layer_out = tf.layers.dense(inputs=inv_latent_layer_2, units=self.state_vector_size, kernel_initializer=tf.zeros_initializer())

        return tf.layers.flatten(inputs=inv_latent_layer_out)

    def predict(self, state, action):
        return self._sess.run(self._inv_latent_nstate, {self._state: state, self._action: action})

    def update(self, state, action, _next_state):
        # self._sess.run(self._train_op_latent, {self._state: state, self._action: action, self._next_state: next_state})
        # self._sess.run(self._train_op_invlatent, {self._state: state, self._action: action, self._next_state: next_state})
        # self._sess.run(self._train_op_forward, {self._state: state, self._action: action, self._next_state: next_state})

        self._sess.run(self._train_op, {self._state: state, self._action: action, self._next_state: next_state})

    def get_all_losses(self, state, action, _next_state):

        loss_latent = self._sess.run(self._loss_latent, {self._state: state, self._action: action, self._next_state: next_state})
        loss_invlatent = self._sess.run(self._loss_invlatent, {self._state: state, self._action: action, self._next_state: next_state})
        loss_forward = self._sess.run(self._loss_forward, {self._state: state, self._action: action, self._next_state: next_state})

        return [loss_latent, loss_invlatent, loss_forward]

    def get_latent_states(self, state, action, _next_state):

        return self._sess.run(self._merged_latent_states, {self._state: state, self._action: action, self._next_state: next_state})

state_size = 160
action_size = 12
latent_size = 32
learning_rate = 0.001

with open('./test_trajectory.npy', 'rb') as f:
    full_trajectory = np.load(f, allow_pickle=True)

num_states = 2*len(full_trajectory)

# from tensorflow.python.framework import ops
tf.reset_default_graph()

with tf.Session() as sess:

    model = Dynamics(sess, state_size, action_size, latent_size, learning_rate)
    
    sess.run(tf.global_variables_initializer())

    #create a file that stores the summary of the operation
    writer = tf.summary.FileWriter("./output", sess.graph)
    #run the session

    i = 0

    for i in range(num_states):

        state_trajectory = full_trajectory[i%600]

        state = state_trajectory['state']
        action = state_trajectory['action'].reshape((1,action_size))
        next_state = state_trajectory['next_state']

        model.update(state, action, next_state)
        print(model.get_all_losses(state, action, next_state))
        i += 1

    results = model.predict(state, action)

    print(model.get_latent_states(state, action, next_state))
    print(model.predict(full_trajectory[10]['state'], full_trajectory[10]['action'].reshape((1,action_size))) - full_trajectory[10]['next_state'])

# print(model.predict(full_trajectory[10]['state'], full_trajectory[10]['action']) - full_trajectory[10]['next_state'])
import numpy as np 
import tensorflow as tf
import gym
from datetime import datetime
import time
import pdb

# tf.enable_eager_execution()

def mlp(x, hidden_layers, output_size, activation=tf.nn.relu, last_activation=None):
    '''
    Multi-layer perceptron
    '''
    inputt = tf.keras.Input(shape=(None, x[0]))
    x = inputt
    for l in hidden_layers:
        x = tf.keras.layers.Dense(units=l, activation=activation)(x)
    output = tf.keras.layers.Dense(units=output_size, activation=last_activation)(x)
    return tf.keras.Model(inputs=inputt, outputs=output)

def softmax_entropy(logits):
    '''
    Softmax Entropy 
    '''
    # tf.print(logits)
    return tf.reduce_sum(tf.nn.softmax(logits, axis=-1) * tf.nn.log_softmax(logits, axis=-1), axis=-1)


def discounted_rewards(rews, gamma):
    '''
    Discounted reward to go 

    Parameters:
    ----------
    rews: list of rewards
    gamma: discount value 
    '''
    rtg = np.zeros_like(rews, dtype=np.float32)
    rtg[-1] = rews[-1]
    for i in reversed(range(len(rews)-1)):
        rtg[i] = rews[i] + gamma*rtg[i+1]
    return rtg

class Buffer():
    '''
    Buffer class to store the experience from a unique policy
    '''
    def __init__(self, gamma=0.99):
        self.gamma = gamma
        self.obs = []
        self.act = []
        self.ret = []

    def store(self, temp_traj):
        '''
        Add temp_traj values to the buffers and compute the advantage and reward to go

        Parameters:
        -----------
        temp_traj: list where each element is a list that contains: observation, reward, action, state-value
        '''
        # store only if the temp_traj list is not empty
        if len(temp_traj) > 0:
            self.obs.extend(temp_traj[:,0])
            rtg = discounted_rewards(temp_traj[:,1], self.gamma)
            self.ret.extend(rtg)
            self.act.extend(temp_traj[:,2])

    def get_batch(self):
        b_ret = self.ret
        return self.obs, self.act, b_ret

    def __len__(self):
        assert(len(self.obs) == len(self.act) == len(self.ret))
        return len(self.obs)
    

def REINFORCE(env_name, hidden_sizes=[32], lr=5e-3, num_epochs=50, gamma=0.99, steps_per_epoch=100):
    '''
    REINFORCE Algorithm

    Parameters:
    -----------
    env_name: Name of the environment
    hidden_size: list of the number of hidden units for each layer
    lr: policy learning rate
    gamma: discount factor
    steps_per_epoch: number of steps per epoch
    num_epochs: number train epochs (Note: they aren't properly epochs)
    '''

    env = gym.make(env_name)    

    optimizer = tf.keras.optimizers.Adam(lr)
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.n 

    ##################################################
    ########### COMPUTE THE LOSS FUNCTIONS ###########
    ##################################################

    # Time
    now = datetime.now()
    clock_time = "{}_{}.{}.{}".format(now.day, now.hour, now.minute, now.second)
    print('Time:', clock_time)

    hyp_str = '-steps_{}-aclr_{}'.format(steps_per_epoch, lr)
    file_writer = tf.summary.create_file_writer('log_dir/{}/REINFORCE_{}_{}'.format(env_name, clock_time, hyp_str))
    
    
    # few variables
    step_count = 0
    train_rewards = []
    train_ep_len = []
    timer = time.time()

    buffer = Buffer(gamma)
    # collect the episodes' information
    obs_batch, act_batch, ret_batch = buffer.get_batch()

    # p_loss = -tf.reduce_mean(p_log*ret_batch)
    # tf.summary.scalar('old_p_loss', p_loss, collections=['pre_train'])
    # pre_scalar_summary = tf.summary.merge_all('pre_train')

    # # run pre_scalar_summary before the optimization phase
    # epochs_summary = pre_scalar_summary(obs_batch, act_batch, ret_batch)
    # file_writer.add_summary(epochs_summary, step_count)

    # policy
    p_logits = mlp(obs_dim, hidden_sizes, act_dim, activation=tf.tanh)
    # tf.print(p_logits)

    # main cycle
    for ep in range(num_epochs):

        # initialize environment for the new epochs
        obs = env.reset()
        # env.render()
        obs = tf.Variable(obs)
        obs = tf.expand_dims(obs, 0)
        # intiaizlie buffer and other variables for the new epochs
        buffer = Buffer(gamma)
        env_buf = []
        ep_rews = []
        while len(buffer) < steps_per_epoch:
            # run the policy
            # print(obs)
            act = tf.squeeze(tf.random.categorical(p_logits(obs), 1))
            # take a step in the environment
            obs2, rew, done, _ = env.step(tf.squeeze(act).numpy())
            # env.render()
            # add the new transition
            env_buf.append([obs.numpy().squeeze(), rew, act.numpy()])
            # print(1)
            obs = tf.expand_dims(obs2, 0)

            step_count += 1
            ep_rews.append(rew)

            if done:
                # store the trajectory just completed
                buffer.store(np.array(env_buf))
                env_buf = []
                # store additionl information about the episode
                train_rewards.append(np.sum(ep_rews))
                train_ep_len.append(len(ep_rews))
                # reset the environment
                obs = env.reset()
                env.render()
                obs = tf.Variable(obs)
                obs = tf.expand_dims(obs, 0)
                # pdb.set_trace()
                ep_rews = []

        # collect the episodes' information
        obs_batch, act_batch, ret_batch = buffer.get_batch()
        
        # policy
        with tf.GradientTape() as tape:
            actions_mask = tf.one_hot(act_batch, depth=act_dim)
            obs_batch = tf.Variable(obs_batch)
            # pdb.set_trace()
            
            # obs_batch = tf.expand_dims(obs_batch, 0)
            p_log = tf.reduce_sum(tf.multiply(actions_mask, tf.nn.log_softmax(p_logits(obs_batch))), axis=1)

            entropy = -tf.reduce_mean(softmax_entropy(p_logits(obs_batch)))
            p_loss = -tf.reduce_mean(p_log*ret_batch)

        gradients = tape.gradient(p_loss,  p_logits.trainable_weights)
        optimizer.apply_gradients(zip(gradients, p_logits.trainable_variables))



        # Set scalars and hisograms for TensorBoard
        # tf.summary.scalar('p_loss', p_loss, collections=['train'])
        # tf.summary.scalar('entropy', entropy, collections=['train'])
        # tf.summary.histogram('p_soft', tf.nn.softmax(p_logits), collections=['train'])
        # tf.summary.histogram('p_log', p_log, collections=['train'])
        # tf.summary.histogram('act_multn', act_multn, collections=['train'])
        # tf.summary.histogram('p_logits', p_logits, collections=['train'])
        # tf.summary.histogram('ret_ph', ret_batch, collections=['train'])
        # train_summary_run = tf.summary.merge_all('train')
        
        # tf.summary.scalar('old_p_loss', p_loss, collections=['pre_train'])
        # pre_scalar_summary = tf.summary.merge_all('pre_train')
        # run train_summary to save the summary after the optimization
        # train_summary_run = sess.run(train_summary, feed_dict={obs_ph:obs_batch, act_ph:act_batch, ret_ph:ret_batch})
        # file_writer.add_summary(train_summary_run, step_count)

        # it's time to print some useful information
        if ep % 10 == 0:
            print('Ep:%d MnRew:%.2f MxRew:%.1f EpLen:%.1f Buffer:%d -- Step:%d -- Time:%d' % (ep, np.mean(train_rewards), np.max(train_rewards), np.mean(train_ep_len), len(buffer), step_count,time.time()-timer))

            # summary = tf.Summary()
            # summary.value.add(tag='supplementary/len', simple_value=np.mean(train_ep_len))
            # summary.value.add(tag='supplementary/train_rew', simple_value=np.mean(train_rewards))
            # file_writer.add_summary(summary, step_count)
            # file_writer.flush()

            timer = time.time()
            train_rewards = []
            train_ep_len = []


    env.close()
    file_writer.close()


if __name__ == '__main__':
    REINFORCE('LunarLander-v2', hidden_sizes=[64], lr=8e-3, gamma=0.99, num_epochs=1000, steps_per_epoch=1000)
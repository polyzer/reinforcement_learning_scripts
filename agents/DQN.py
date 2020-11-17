import numpy as np
import tensorflow as tf
import gym
from datetime import datetime
from collections import deque
import time
import sys
# from atari_wrappers import make_env

tf.config.set_visible_devices([], 'GPU')

current_milli_time = lambda: int(round(time.time() * 1000))

def cnn(x):
    x = tf.layers.conv2d(x, filters=16, kernel_size=8, strides=4, padding='valid', activation='relu')
    x = tf.layers.conv2d(x, filters=32, kernel_size=4, strides=2, adding='valid', activation='relu')
    return tf.layers.conv2d(x, filters=32, kernel_size=3, strides=1, padding='valid', activation='relu')
    
def fnn(x, hidden_layers, output_layer, activation=tf.nn.relu, last_activation=None):
    for l in hidden_layers:
        x = tf.layers.dense(x, units=l, activation=activation)
    return tf.layers.dense(x, units=output_layer, activation=last_activation)

def qnet(x, hidden_layers, output_size, fnn_activation=tf.nn.relu, last_activation=None):
    x = cnn(x)
    x = tf.layers.flatten(x)
    return fnn(x, hidden_layers, output_size, fnn_activation, last_activation)

def mlp(x, hidden_layers, output_layer, activation=tf.tanh, last_activation=None):
    '''
    Multi-layer perceptron
    '''
    for l in hidden_layers:
        x = tf.layers.dense(x, units=l, activation=activation)
    return tf.layers.dense(x, units=output_layer, activation=last_activation)


def greedy(action_values):
    '''
    Greedy policy
    '''
    return np.argmax(action_values)

def eps_greedy(action_values, eps=0.1):
    '''
    Eps-greedy policy
    '''
    if np.random.uniform(0,1) < eps:
        # Choose a uniform random action
        return np.random.randint(len(action_values))
    else:
        # Choose the greedy action
        return np.argmax(action_values)

def test_agent(env_test, agent_op, num_games=20):
    '''
    Test an agent
    '''
    games_r = []

    for _ in range(num_games):
        d = False
        game_r = 0
        o = env_test.reset()

        while not d:
            # Use an eps-greedy policy with eps=0.05 (to add stochasticity to the policy)
            # Needed because Atari envs are deterministic
            # If you would use a greedy policy, the results will be always the same
            a = eps_greedy(np.squeeze(agent_op(o)), eps=0.05)
            o, r, d, _ = env_test.step(a)

            game_r += r

        games_r.append(game_r)

    return games_r


def q_target_values(mini_batch_rw, mini_batch_done, av, discounted_value):   
    '''
    Calculate the target value y for each transition
    '''
    max_av = np.max(av, axis=1)
    
    # if episode terminate, y take value r
    # otherwise, q-learning step
    
    ys = []
    for r, d, av in zip(mini_batch_rw, mini_batch_done, max_av):
        if d:
            ys.append(r)
        else:
            q_step = r + discounted_value * av
            ys.append(q_step)
    
    assert len(ys) == len(mini_batch_rw)
    return ys

class StructEnv(gym.Wrapper):
    '''
    Gym Wrapper to store information like number of steps and total reward of the last espisode.
    '''
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.n_obs = self.env.reset()
        self.rew_episode = 0
        self.len_episode = 0

    def reset(self, **kwargs):
        self.n_obs = self.env.reset(**kwargs)
        self.rew_episode = 0
        self.len_episode = 0
        return self.n_obs.copy()
        
    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.rew_episode += reward
        self.len_episode += 1
        return ob, reward, done, info

    def get_episode_reward(self):
        return self.rew_episode

    def get_episode_length(self):
        return self.len_episode


class ExperienceBuffer():
    def __init__(self, buffer_size):
        self.obs_buf = deque(maxlen=buffer_size)
        self.rew_buf = deque(maxlen=buffer_size)
        self.act_buf = deque(maxlen=buffer_size)
        self.obs2_buf = deque(maxlen=buffer_size)
        self.done_buf = deque(maxlen=buffer_size)
    def add(self, obs, rew, act, obs2, done):
        self.obs_buf.append(obs)
        self.rew_buf.append(rew)
        self.act_buf.append(act)
        self.obs2_buf.append(obs2)
        self.done_buf.append(done)

    def sample_minibatch(self, batch_size):
        mb_indices = np.random.randint(len(self.obs_buf), size=batch_size)
        mb_obs = scale_frames([self.obs_buf[i] for i in mb_indices])
        mb_rew = [self.rew_buf[i] for i in mb_indices]
        mb_act = [self.act_buf[i] for i in mb_indices]
        mb_obs2 = scale_frames([self.obs2_buf[i] for i in mb_indices])
        mb_done = [self.done_buf[i] for i in mb_indices]
        return mb_obs, mb_rew, mb_act, mb_obs2, mb_done
        
    def __len__(self):
        return len(self.obs_buf)

def DQN(env_name, hidden_sizes=[32], lr=1e-2, num_epochs=2000,
        buffer_size=100000, discount=0.99, update_target_net=1000, batch_size=64,
        update_freq=4, frames_num=2, min_buffer_size=5000, test_frequency=20,
        start_explor=1, end_explor=0.1, explor_steps=100000):
    env = gym.make(env_name)
    env_test = gym.make(env_name)
    # env_test = gym.wrappers.Monitor(env_test,"VIDEOS/TEST_VIDEOS"+env_name+str(current_milli_time()),force=True,video_callable=lambda x: x%20==0)
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.n
    # act_ph = tf.placeholder(shape=(None,), dtype=tf.int32, name='act')
    # y_ph = tf.placeholder(shape=(None,), dtype=tf.float32, name='y')

    with tf.variable_scope('target_network'):
        target_qv = mlp(obs_dim[0], hidden_sizes, act_dim)
        target_vars = tf.trainable_variables()
    with tf.variable_scope('online_network'):
        # online_qv = qnet(obs_ph, hidden_sizes, act_dim)
        online_qv = mlp(obs_dim[0], hidden_sizes, act_dim)
        train_vars = tf.trainable_variables()
        update_target = [train_vars[i].assign(train_vars[i+len(target_vars)]) for i in range(len(train_vars) - len(target_vars))]
        update_target_op = tf.group(*update_target)

    act_onehot = tf.one_hot(act_ph, depth=act_dim)
    q_values = tf.reduce_sum(act_onehot * online_qv, axis=1)
    # MSE loss function
    v_loss = tf.reduce_mean((y_ph - q_values)**2)
    # Adam optimize that minimize the loss v_loss
    v_opt = tf.train.AdamOptimizer(lr).minimize(v_loss)


    now = datetime.now()
    clock_time = "{}_{}.{}.{}".format(now.day, now.hour, now.minute,
    int(now.second))
    mr_v = tf.Variable(0.0)
    ml_v = tf.Variable(0.0)
    # tf.summary.scalar('v_loss', v_loss)
    # tf.summary.scalar('Q-value', tf.reduce_mean(q_values))
    # tf.summary.histogram('Q-values', q_values)
    # scalar_summary = tf.summary.merge_all()
    # reward_summary = tf.summary.scalar('test_rew', mr_v)
    # mean_loss_summary = tf.summary.scalar('mean_loss', ml_v)
    hyp_str = "-lr_{}-upTN_{}-upF_{}-frms_{}".format(lr, update_target_net, update_freq, frames_num)
    # file_writer =tf.summary.FileWriter('log_dir/'+env_name+'/DQN_'+clock_time+'_'+hyp_str,tf.get_default_graph())
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    step_count = 0
    last_update_loss = []
    ep_time = current_milli_time()

    batch_rew = []
    obs = env.reset()  

    obs = env.reset()
    buffer = ExperienceBuffer(buffer_size)
    sess.run(update_target_op)
    eps = start_explor
    eps_decay = (start_explor - end_explor) / explor_steps

    for ep in range(num_epochs):
        g_rew = 0
        done = False
        while not done:
            act = eps_greedy(np.squeeze(agent_op(obs)), eps=eps)
            obs2, rew, done, _ = env.step(act)
            buffer.add(obs, rew, act, obs2, done)
            obs = obs2
            g_rew += rew
            step_count += 1

            if eps > end_explor:
                eps -= eps_decay

            if len(buffer) > min_buffer_size and (step_count % update_freq == 0):
                mb_obs, mb_rew, mb_act, mb_obs2, mb_done = buffer.sample_minibatch(batch_size)
                mb_trg_qv = target_qv(mb_obs2)
                y_r = q_target_values(mb_rew, mb_done, mb_trg_qv, discount)
                # Compute the target values
                train_summary, train_loss, _ = sess.run([scalar_summary, v_loss, v_opt], feed_dict={obs_ph:mb_obs, y_ph:y_r, act_ph: mb_act})
                tf.train.AdamOptimizer(lr).minimize(v_loss)
                file_writer.add_summary(train_summary, step_count)
                last_update_loss.append(train_loss)

            if (len(buffer) > min_buffer_size) and (step_count % update_target_net) == 0:
                _, train_summary = sess.run([update_target_op, mean_loss_summary], feed_dict={ml_v:np.mean(last_update_loss)})
                file_writer.add_summary(train_summary, step_count)
                last_update_loss = []
            
            if done:
                obs = env.reset()
                batch_rew.append(g_rew)
                g_rew = 0

        if ep % test_frequency == 0:
            test_rw = test_agent(env_test, agent_op, num_games=10)
            test_summary = sess.run(reward_summary, feed_dict={mr_v:np.mean(test_rw)})
            file_writer.add_summary(test_summary, step_count)
            print('Ep:%4d Rew:%4.2f, Eps:%2.2f -- Step:%5d -- Test:%4.2f %4.2f' % (ep,np.mean(batch_rew), eps, step_count, np.mean(test_rw), np.std(test_rw)))
            batch_rew = []
    file_writer.close()
    env.close()
    env_test.close()

if __name__ == '__main__':
    DQN('Acrobot-v1', hidden_sizes=[128], lr=2e-4, buffer_size=100000, update_target_net=1000, batch_size=32, update_freq=2, frames_num=2, min_buffer_size=10000)
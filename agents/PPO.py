import sys
import os
sys.path.append(os.path.abspath("./"))
from envs.TestEnv import TestEnv
import numpy as np 
import tensorflow as tf
import gym
from datetime import datetime
import time
import pybullet
import pybullet_envs
import pdb
import logging
tf.get_logger().setLevel(logging.ERROR)
# tf.config.set_visible_devices([], 'GPU')

def mlp(x, hidden_layers, output_size, activation=tf.nn.relu, last_activation=None):
    '''
    Multi-layer perceptron
    '''
    inputt = tf.keras.Input(shape=(None, x[0]))
    x = inputt
    for l in hidden_layers:
        x = tf.keras.layers.Dense(units=l, activation=activation,)(x)
#        x = tf.keras.layers.Dense(units=l, activation=activation, kernel_initializer=tf.keras.initializers.Zeros())(x)
    # output = tf.keras.layers.Dense(units=output_size, activation=last_activation, kernel_initializer=tf.keras.initializers.Zeros())(x)
    output = tf.keras.layers.Dense(units=output_size, activation=last_activation,)(x)
    return tf.keras.Model(inputs=inputt, outputs=output)

def softmax_entropy(logits):
    '''
    Softmax Entropy
    '''
    return -tf.reduce_sum(tf.nn.softmax(logits, axis=-1) * tf.nn.log_softmax(logits, axis=-1), axis=-1)

def clipped_surrogate_obj(new_p, old_p, adv, eps):
    '''
    Clipped surrogate objective function
    '''
    rt = tf.exp(new_p - old_p) # i.e. pi / old_pi
    ret = -tf.reduce_mean(tf.minimum(rt*adv, tf.clip_by_value(rt, 1-eps, 1+eps)*adv))
    pdb.set_trace()
    return ret

def GAE(rews, v, v_last, gamma=0.99, lam=0.95):
    '''
    Generalized Advantage Estimation
    '''
    assert len(rews) == len(v)

    vs = np.append(v, v_last)
    delta = np.array(rews) + gamma*vs[1:] - vs[:-1]
    gae_advantage = discounted_rewards(delta, 0, gamma*lam)
    return gae_advantage

def discounted_rewards(rews, last_sv, gamma):
    '''
    Discounted reward to go 

    Parameters:
    ----------
    rews: list of rewards
    last_sv: value of the last state
    gamma: discount value 
    '''
    rtg = np.zeros_like(rews, dtype=np.float32)
    rtg[-1] = rews[-1] + gamma*last_sv
    for i in reversed(range(len(rews)-1)):
        rtg[i] = rews[i] + gamma*rtg[i+1]
    return rtg



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

class Buffer():
    '''
    Class to store the experience from a unique policy
    '''
    def __init__(self, gamma=0.99, lam=0.95):
        self.gamma = gamma
        self.lam = lam
        self.adv = []
        self.ob = []
        self.ac = []
        self.rtg = []

    def store(self, temp_traj, last_sv):
        '''
        Add temp_traj values to the buffers and compute the advantage and reward to go

        Parameters:
        -----------
        temp_traj: list where each element is a list that contains: observation, reward, action, state-value
        last_sv: value of the last state (Used to Bootstrap)
        '''
        # store only if there are temporary trajectories
        if len(temp_traj) > 0:
            # pdb.set_trace()
            self.ob.extend(temp_traj[:,0])
            rtg = discounted_rewards(temp_traj[:,1], last_sv, self.gamma)
            self.adv.extend(GAE(temp_traj[:,1], temp_traj[:,3], last_sv, self.gamma, self.lam))
            self.rtg.extend(rtg)
            self.ac.extend(temp_traj[:,2])


    def get_batch(self):
        # standardize the advantage values
        norm_adv = (self.adv - np.mean(self.adv)) / (np.std(self.adv) + 1e-10)
        return np.array(self.ob), np.array(self.ac), np.array(norm_adv), np.array(self.rtg)

    def __len__(self):
        assert(len(self.adv) == len(self.ob) == len(self.ac) == len(self.rtg))
        return len(self.ob)
    
def gaussian_log_likelihood(x, mean, log_std):
    '''
    Gaussian Log Likelihood 
    '''
    log_p = -0.5 *((x-mean)**2 / (tf.exp(log_std)**2+1e-9) + 2*log_std + np.log(2*np.pi))
    return tf.reduce_sum(log_p, axis=-1)

def PPO(env_name, hidden_sizes=[32], cr_lr=5e-3, ac_lr=5e-3, num_epochs=50, minibatch_size=5000, gamma=0.99, lam=0.95, number_envs=1, eps=0.1, 
        actor_iter=5, critic_iter=10, steps_per_env=100, action_type='Discrete'):
    '''
    Proximal Policy Optimization

    Parameters:
    -----------
    env_name: Name of the environment
    hidden_size: list of the number of hidden units for each layer
    ac_lr: actor learning rate
    cr_lr: critic learning rate
    num_epochs: number of training epochs
    minibatch_size: Batch size used to train the critic and actor
    gamma: discount factor
    lam: lambda parameter for computing the GAE
    number_envs: number of parallel synchronous environments
        # NB: it isn't distributed across multiple CPUs
    eps: Clip threshold. Max deviation from previous policy.
    actor_iter: Number of SGD iterations on the actor per epoch
    critic_iter: NUmber of SGD iterations on the critic per epoch
    steps_per_env: number of steps per environment
            # NB: the total number of steps per epoch will be: steps_per_env*number_envs
    action_type: class name of the action space: Either "Discrete' or "Box"
    '''

    def act_smp_discrete(p_logits):
        return tf.squeeze(tf.random.multinomial(p_logits, 1))

    def act_smp_cont(p_noisy, low_action_space, high_action_space):
        ret = tf.clip_by_value(p_noisy, low_action_space, high_action_space)
        return ret

    def get_p_noisy(p_logits, log_std):
        ret = p_logits + tf.random.normal(tf.shape(p_logits), 0, 1) * tf.exp(log_std)
        return ret

    def get_p_log_discrete(p_logits, act_ph, act_dim, log_std):
        act_onehot = tf.one_hot(act_ph, depth=act_dim)
        return tf.reduce_sum(act_onehot * tf.nn.log_softmax(p_logits), axis=-1)

    def get_p_log_cont(x, mean, log_std):
        ret = gaussian_log_likelihood(x, mean, log_std)
        return ret

    # Create some environments to collect the trajectories
    
    # envs = [StructEnv(TestEnv()) for _ in range(number_envs)]
    envs = [StructEnv(gym.make(env_name)) for _ in range(number_envs)]
    obs_dim = envs[0].observation_space.shape

    # Placeholders
    if action_type == 'Discrete':
        act_dim = envs[0].action_space.n 

    elif action_type == 'Box':
        low_action_space = envs[0].action_space.low
        high_action_space = envs[0].action_space.high
        act_dim = envs[0].action_space.shape[0]


    # Computational graph for the policy in case of a continuous action space
    if action_type == 'Discrete':
        p_logits = mlp(obs_dim, hidden_sizes, act_dim, tf.nn.relu, last_activation=tf.tanh)
        act_smp = act_smp_discrete
        p_log = get_p_log_discrete
        
    # Computational graph for the policy in case of a continuous action space
    else:
        p_logits = mlp(obs_dim, hidden_sizes, act_dim, tf.tanh, last_activation=tf.tanh)
        log_std = tf.Variable(np.zeros(act_dim, dtype=np.float32)-0.5, name='log_std')
        
        # Add noise to the mean values predicted
        # The noise is proportional to the standard deviation
        p_noisy = get_p_noisy
        # Clip the noisy actions
        act_smp = act_smp_cont
        # Compute the gaussian log likelihood
        p_log = get_p_log_cont

    # Nerual nework value function approximizer
    s_values = mlp(obs_dim, hidden_sizes, 1, tf.tanh, last_activation=None)


    # PPO loss function
    
    # MSE loss function

    # policy optimizer
    p_opt = tf.keras.optimizers.Adam(ac_lr)
    # value function optimizer
    v_opt = tf.keras.optimizers.Adam(cr_lr)

    # Time
    now = datetime.now()
    clock_time = "{}_{}.{}.{}".format(now.day, now.hour, now.minute, now.second)
    print('Time:', clock_time)

    # Set scalars and hisograms for TensorBoard
    # tf.summary.scalar('p_loss', p_loss, collections=['train'])
    # tf.summary.scalar('v_loss', v_loss, collections=['train'])
    # tf.summary.scalar('s_values_m', tf.reduce_mean(s_values), collections=['train'])

    # if action_type == 'Box':
    #     tf.summary.scalar('p_std', tf.reduce_mean(tf.exp(log_std)), collections=['train'])
    #     tf.summary.histogram('log_std',log_std, collections=['train'])
    # tf.summary.histogram('p_log', p_log, collections=['train'])
    # tf.summary.histogram('p_logits', p_logits, collections=['train'])
    # tf.summary.histogram('s_values', s_values, collections=['train'])
    # tf.summary.histogram('adv_ph',adv_ph, collections=['train'])
    # scalar_summary = tf.summary.merge_all('train')

    # .. summary to run before the optimization steps
    # tf.summary.scalar('old_v_loss', v_loss, collections=['pre_train'])
    # tf.summary.scalar('old_p_loss', p_loss, collections=['pre_train'])
    # pre_scalar_summary = tf.summary.merge_all('pre_train')

    hyp_str = '-bs_'+str(minibatch_size)+'-envs_'+str(number_envs)+'-ac_lr_'+str(ac_lr)+'-cr_lr'+str(cr_lr)+'-act_it_'+str(actor_iter)+'-crit_it_'+str(critic_iter)
    file_writer = tf.summary.create_file_writer('log_dir/'+env_name+'/PPO_'+clock_time+'_'+hyp_str)
    
    
    # variable to store the total number of steps
    step_count = 0
    
    print('Env batch size:',steps_per_env, ' Batch size:',steps_per_env*number_envs)

    for ep in range(num_epochs):
        # Create the buffer that will contain the trajectories (full or partial) 
        # run with the last policy
        buffer = Buffer(gamma, lam)
        # lists to store rewards and length of the trajectories completed
        batch_rew = []
        batch_len = []
        # Execute in serial the environments, storing temporarily the trajectories. 
        for env in envs:
            temp_buf = []
            # pdb.set_trace()
            #iterate over a fixed number of steps
            for k in range(steps_per_env):
                # print(k)
                # run the policy
                nobs = tf.Variable([env.n_obs])
                nobs = tf.expand_dims(nobs, 0)
                p_logits_val = p_logits(nobs)
                p_noisy_val = p_noisy(p_logits_val, log_std)
                act = act_smp(p_noisy_val, low_action_space, high_action_space)
                val = s_values(nobs)
                act = np.squeeze(act)
                
                # take a step in the environment
                obs2, rew, done, _ = env.step(act)
                # add the new transition to the temporary buffer
                temp_buf.append([env.n_obs.copy(), rew, act, np.squeeze(val)])

                env.n_obs = obs2.copy()
                step_count += 1

                if done:
                    # Store the full trajectory in the buffer 
                    # (the value of the last state is 0 as the trajectory is completed)
                    buffer.store(np.array(temp_buf), 0)

                    # Empty temporary buffer
                    temp_buf = []
                    
                    batch_rew.append(env.get_episode_reward())
                    batch_len.append(env.get_episode_length())
                    
                    # reset the environment
                    env.render()
                    env.reset()                 

            # Bootstrap with the estimated state value of the next state!
            nobs = tf.Variable([env.n_obs])
            nobs = tf.expand_dims(nobs, 0)
            last_v = s_values(nobs)
            buffer.store(np.array(temp_buf), np.squeeze(last_v))


        # Gather the entire batch from the buffer
        # NB: all the batch is used and deleted after the optimization. That is because PPO is on-policy
        obs_batch, act_batch, adv_batch, rtg_batch = buffer.get_batch()
        old_p_log = None
        if action_type == "Box":
            old_p_log = p_log(act_batch, p_logits(obs_batch),log_std)
        old_p_batch = np.array(old_p_log)

        # summary = pre_scalar_summary(obs_batch, act_batch, adv_batch, rtg_batch, old_p_batch)
        # file_writer.add_summary(summary, step_count)

        lb = len(buffer)
        shuffled_batch = np.arange(lb)    
        # print("IM here")
        # Policy optimization steps
        for _ in range(actor_iter):
            # shuffle the batch on every iteration
            np.random.shuffle(shuffled_batch)
            for idx in range(0,lb, minibatch_size):
                # pdb.set_trace()
                minib = shuffled_batch[idx:min(idx+minibatch_size,lb)]
                old_p_log_ph = old_p_batch[minib]
                with tf.GradientTape() as tape:
                    p_logits_value = p_logits(obs_batch[minib])
                    p_log_value = p_log( act_batch[minib], p_logits_value, log_std)
                    p_loss = clipped_surrogate_obj(p_log_value, old_p_batch[minib], adv_batch[minib], eps)
                
                gradients = tape.gradient(p_loss,  p_logits.trainable_weights)
                p_opt.apply_gradients(zip(gradients, p_logits.trainable_variables))
                # p_opt(obs_batch[minib], act_batch[minib], adv_batch[minib], old_p_batch[minib])
                    
        # Value function optimization steps
        for _ in range(critic_iter):
            # shuffle the batch on every iteration
            np.random.shuffle(shuffled_batch)
            for idx in range(0,lb, minibatch_size):
                minib = shuffled_batch[idx:min(idx+minibatch_size,lb)]
                with tf.GradientTape() as tape:
                    s_v = tf.squeeze(s_values(obs_batch[minib]))
                    v_loss = tf.reduce_mean((rtg_batch[minib] - s_v)**2)
                gradients = tape.gradient(v_loss,  s_values.trainable_weights)
                v_opt.apply_gradients(zip(gradients, s_values.trainable_variables))


        # print some statistics and run the summary for visualizing it on TB
        if len(batch_rew) > 0:           
            # train_summary = sess.run(scalar_summary, feed_dict={obs_ph:obs_batch, act_ph:act_batch, adv_ph:adv_batch, 
            #                                                     old_p_log_ph:old_p_batch, ret_ph:rtg_batch})
            # file_writer.add_summary(train_summary, step_count)

            # summary = tf.Summary()
            # summary.value.add(tag='supplementary/performance', simple_value=np.mean(batch_rew))
            # summary.value.add(tag='supplementary/len', simple_value=np.mean(batch_len))
            # file_writer.add_summary(summary, step_count)
            # file_writer.flush()
        
            print('Ep:%d Rew:%.2f -- Step:%d' % (ep, np.mean(batch_rew), step_count))

    # closing environments..
    for env in envs:
        env.close()

    # Close the writer
    file_writer.close()


if __name__ == '__main__':
    PPO('Walker2DBulletEnv-v0', hidden_sizes=[64,64], cr_lr=5e-4, ac_lr=2e-4, gamma=0.99, lam=0.95, steps_per_env=1000, 
        number_envs=1, eps=0.15, actor_iter=6, critic_iter=10, action_type='Box', num_epochs=5000, minibatch_size=256)
      
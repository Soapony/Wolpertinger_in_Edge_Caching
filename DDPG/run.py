import numpy as np
from collections import deque
import tensorflow as tf
import keras.backend as K

from ddpg import DDPG
from wolpertinger import Wolpertinger

from cache_env import cache_env
from gen_zipf import gen_zipf



def main(low_list,
         high_list,
         pts_list,
         b_wolpertinger):
    sess = tf.Session()
    K.set_session(sess)

    zipf = gen_zipf(1.3, 10000, 5000,False)
    requests_list = zipf.load_request("training_data.txt")
    # Define environment
    env = cache_env(300, requests_list, False, 0.6)

    if b_wolpertinger:
        ddpg = Wolpertinger(env, sess, low_list=low_list, high_list=high_list, points_list=pts_list)
    else:
        ddpg = DDPG(env, sess, low_action_bound_list=low_list, high_action_bound_list=high_list)

    # Main loop
    num_episodes = 5

    current_state, done = env.reset()
    episodes = 0

    reward_error=[]
    predict_rewards=[]
    actual_rewards=[]

    while(episodes < num_episodes):
        
        if done:
            print("hit rate: ",env.get_hit_rate())
            done = False
            episodes +=1
            current_state, done = env.reset()
            continue

        current_state = current_state.reshape((1, ddpg.state_dim))
        action, predict_reward = ddpg.act(episodes, current_state)

        action = action.reshape((1, ddpg.action_dim))
        next_state, reward, done = env.step(action[0][0])
        next_state = next_state.reshape((1, ddpg.state_dim))

        predict_rewards.append(predict_reward)
        actual_rewards.append(reward)
        reward_error.append(predict_reward - reward)
        print('DEBUG ACTION REWARD_ERROR: ', action, predict_reward - reward)
        
        ddpg.replay_buffer.add(current_state, action, reward, next_state, done)
        current_state = next_state

        ddpg.train()
        ddpg.update_target_models()
    
    f1 = open("result/reward_error.txt","w")
    f1.write(str(reward_error)+"\n")
    f1.close()
    f2 = open("result/pre_act_reward.txt","w")
    f2.write(str(predict_rewards)+"\n")
    f2.write(str(actual_rewards)+"\n")
    f2.close()

        



if __name__ == '__main__':
    cache_size = 300
    main(low_list=[0], high_list=[cache_size], pts_list=[cache_size], b_wolpertinger=True)


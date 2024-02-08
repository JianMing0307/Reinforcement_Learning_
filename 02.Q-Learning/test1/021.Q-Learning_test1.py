#以下例子的環境是一個一維世界, 在世界的右邊有寶藏(T), 探索者(o)只要找到寶藏就會得到reward, 然後記住如何得到寶藏。

import numpy as np
import pandas as pd
import time

np.random.seed(2)  # reproducible

N_STATES = 6   #----->一維世界寬度
ACTIONS = ['left', 'right']     
EPSILON = 0.9   
ALPHA = 0.1     #----->Learning rate
GAMMA = 0.9     
MAX_EPISODES = 13   
FRESH_TIME = 0.05    #----->移动间隔时间

def build_q_table(n_states, actions):
    table = pd.DataFrame(
        np.zeros((n_states, len(actions))),
        columns=actions,
    )
    print(table)
    return table


def choose_action(state, q_table):     #----->根據Q_table去選擇action
    state_actions = q_table.iloc[state, :]
    if (np.random.uniform() > EPSILON) or (state_actions.all() == 0):
        action_name = np.random.choice(ACTIONS)
    else :
        action_name = state_actions.idxmax()
    return action_name


def get_env_feeback(S, A):
    if A == 'right' :    #----->往右走分2兩種情境:找到寶藏/還沒找到寶藏
        if S == N_STATES-2:
            S_ = 'terminal' #----->S_預設為字串,要用loc
            R = 1
        else : 
            S_ = S + 1
            R = 0
    else :               #----->往左走分兩種情境:在state=0往左走/在state:1~5往左走
        R = 0
        if S == 0:
           S_ = S
        else :
            S_ = S-1
    return S_, R 

def update_env(S, episode, step_counter):
    # This is how environment be updated
    env_list = ['-']*(N_STATES-1) + ['T']   # '---------T' our environment
    if S == 'terminal':
        interaction = 'Episode %s: total_steps = %s' % (episode+1, step_counter)
        print('\r{}'.format(interaction), end='')
        time.sleep(2)
        print('\r                                ', end='')
    else:
        env_list[S] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(FRESH_TIME)

def rl():
    q_table = build_q_table(N_STATES, ACTIONS)
    for episode in range(MAX_EPISODES):
        step_counter = 0
        S = 0
        is_terminated = False 
        update_env(S, episode, step_counter)
        while not is_terminated:


            A = choose_action(S, q_table)
            S_, R = get_env_feeback(S, A)
            q_predict = q_table.loc[S, A] #----->估算值
            if S_ != 'terminal' : 
                q_target = R + GAMMA * q_table.iloc[S_, :].max()
            else:
                q_target = R     #  实际的(状态-行为)值 (回合结束)
                is_terminated = True    # terminate this episode

            q_table.loc[S, A] += ALPHA * (q_target - q_predict)  #----->q_table 更新
            S = S_  

            update_env(S, episode, step_counter+1)  

            step_counter += 1
    return q_table
    
if __name__ == "__main__":
    q_table = rl()
    print('\r\nQ-table:\n')
    print(q_table)
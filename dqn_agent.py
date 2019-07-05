# coding: utf-8

import gym
import numpy as np
from tensorflow.keras.layers import Input, Dense, Flatten, BatchNormalization
from tensorflow.keras.models import Model, clone_model
from tensorflow.keras.optimizers import Adam
from tensorflow.losses import huber_loss

from collections import deque



class ReplayMemory:
    '''experience replay'''
    def __init__(self, memory_size=100000):
        self.memory_size = memory_size
        self.memory = deque([], maxlen = memory_size)

    def append(self, experience):
        '''store experience'''
        self.memory.append(experience)

    def initialize(self, env, initial_memory_size=1000):
        '''事前に経験を蓄える'''
        step = 0
        while True:
            state = env.reset()
            done = False
            while not done:
                action = env.action_space.sample()  # ランダムに行動を選択
                next_state, reward, done, _ = env.step(action)  # 状態、報酬、終了判定の取得

                experience = {
                    'state': state,
                    'next_state': next_state,
                    'reward': reward,
                    'action': action,
                    'done': int(done)
                }
                self.append(experience) # 経験の記憶

                state = next_state
                step += 1

            if step >= initial_memory_size:
                break
        print('pooled experiences for initialization!!')

    def sample(self, batch_size):
        '''sampling experiences in replay buffer'''
        # ランダムにsampling
        batch_indexes = np.random.randint(0, len(self.memory), size=batch_size).tolist()

        state = np.array([self.memory[index]['state'] for index in batch_indexes])
        next_state = np.array([self.memory[index]['next_state'] for index in batch_indexes])
        reward = np.array([self.memory[index]['reward'] for index in batch_indexes])
        action = np.array([self.memory[index]['action'] for index in batch_indexes])
        done = np.array([self.memory[index]['done'] for index in batch_indexes])

        return {'state': state, 'next_state': next_state, 'reward': reward, 'action': action, 'done': done}


class Agent:
    '''policyとnetwork'''
    def __init__(self, action_space, epsilon=0.2):
        self.epsilon = epsilon
        self.actions = list(range(action_space))
        self.model = None
        self.target_model = None

    def q_value(self, state, is_target=False):
        '''Q値を計算'''
        if not is_target:
            model = self.model
        else:
            model = self.target_model
        return model.predict(state)

    def policy(self, state):
        '''epsilon-greedy policy'''
        if np.random.random() < self.epsilon:
            return np.random.randint(len(self.actions))
        else:
            q_value = self.q_value(state)
            return np.argmax(q_value)

    def eps_decay(self, step, eps_end=0.2, n_steps=10000):
        '''stepが進むごとに減衰'''
        eps_start = 1.0
        eps = max(eps_end, (eps_end - eps_start) / n_steps * step + eps_start)
        self.epsilon = eps
        return eps

    def play(self, env, n_episodes=5, render=False):
        '''test'''
        self.epsilon = 0
        memory = deque([])  # ログ用
        for episode in range(n_episodes):
            state = env.reset()
            done = False
            episode_rewards = 0
            while not done:
                if render:
                    env.render()
                action = self.policy(np.array([state]))
                next_state, reward, done, _ = env.step(action)
                experience = {
                    'state': state,
                    'next_state': next_state,
                    'reward': reward,
                    'action': action,
                    'done': int(done)
                }
                episode_rewards += reward
                memory.append(experience)
                state = next_state
            else:
                print('Get reward: {0} after {1} episode '.format(episode_rewards, episode))

        return memory


### environment ###
memory_size = 100000  # メモリーサイズ
initial_memory_size = 1000  # 事前に貯める経験数


env = gym.make('MountainCar-v0')
# env = gym.make('Acrobot-v1')
# env = gym.make('CartPole-v0')

obs_space = env.observation_space.shape[0]
act_space = env.action_space.n

print('obs: {0}, act: {1}'.format(obs_space, act_space))


replay_memory = ReplayMemory(memory_size)
replay_memory.initialize(env, initial_memory_size)  # はじめにある程度経験を蓄えておく


### network ###
lr = 0.0001  # 学習率

state = Input(shape=(obs_space, ))
x = Dense(16, activation='relu', kernel_initializer="he_uniform")(state)
x = BatchNormalization()(x)
x = Dense(16, activation='relu', kernel_initializer="he_uniform")(x)
x = BatchNormalization()(x)
x = Dense(16, activation='relu', kernel_initializer="he_uniform")(x)
x = BatchNormalization()(x)
action = Dense(act_space, kernel_initializer="he_uniform")(x)

model = Model(inputs=state, outputs=action)
model.summary()

model.compile(loss=huber_loss, optimizer=Adam(lr=lr))


### train loop ###
discount_rate = 0.99  # 割引率
eps = 0.1
target_update_interval_steps = 1000  # 重みの更新間隔
batch_size = 32
n_episodes = 1000
total_steps = 0

## Agentを定義
agent = Agent(act_space, epsilon=eps)
agent.model = model  # original network
agent.target_model = clone_model(model)  # target network


for episode in range(n_episodes):
    state = env.reset()
    done = False

    steps_per_epi = 0
    episode_rewards = 0
    episode_q_max = []
    episode_loss = []
    while not done:
        # env.render()
        # Q値のログ
        q_value = agent.q_value(np.array([state]))
        episode_q_max.append(np.max(q_value))

        temp_eps = agent.eps_decay(total_steps, eps_end=eps, n_steps=200*50) # epsilonを減衰
        action = agent.policy(np.array([state]))  # policyにしたがってactionを選択
        next_state, reward, done, _ = env.step(action)

        reward = np.sign(reward)  # 報酬のクリッピング
        episode_rewards += reward  # エピソード内の報酬を更新

        experience = {
                'state': state,
                'next_state': next_state,
                'reward': reward,
                'action': action,
                'done': int(done)
                }
        replay_memory.append(experience)  # 経験の記憶

        # train on batch
        train_batch = replay_memory.sample(batch_size)  # 経験のサンプリング
        q_original = agent.q_value(train_batch['state'])  # オリジナルネットのQ(s,a)
        q_target_next = agent.q_value(train_batch['next_state'], is_target=True)  # ターゲットネットのQ_theta(s',a)

        fixed_q_value = train_batch['reward'] + (1 - train_batch['done']) * discount_rate * np.max(q_target_next, axis=1)  # ベルマン方程式
        for batch_index, action in enumerate(train_batch['action']):
            q_original[batch_index][action] = fixed_q_value[batch_index]  # Q値を更新

        loss = agent.model.train_on_batch(x=train_batch['state'], y=q_original)  # targetネットを教師信号として固定
        episode_loss.append(np.min(loss))

        state = next_state

       # 一定期間ごとにターゲットネットワークの重みを更新
        if (total_steps + 1) % target_update_interval_steps == 0:
            agent.target_model.set_weights(agent.model.get_weights())
            print('target network update!!')

        steps_per_epi += 1
        total_steps += 1

    if (episode + 1) % 10 == 0:
        print('Episode: {}, Reward: {}, Q_max: {:.4f}, loss_min: {:.4f}'.format(episode+1, episode_rewards, np.mean(episode_q_max), np.mean(episode_loss)))
        print('eps: ', temp_eps)
        print('=== test play ===')
        agent.play(env, n_episodes=5, render=False)

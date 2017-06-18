import gym
import numpy as np
import universe  # register the universe environments

env = gym.make('flashgames.DuskDrive-v0')
env.configure(remotes=1)  # automatically creates a local docker container
print("action_space:{}".format(env.action_space))
observation_n = env.reset()

# define our turns or keyboard actions
left = [('KeyEvent', 'ArrowUp', True), ('KeyEvent', 'ArrowLeft', True), ('KeyEvent', 'ArrowRight', False)]
right = [('KeyEvent', 'ArrowUp', True), ('KeyEvent', 'ArrowLeft', False), ('KeyEvent', 'ArrowRight', True)]
forward = [('KeyEvent', 'ArrowUp', True), ('KeyEvent', 'ArrowLeft', False), ('KeyEvent', 'ArrowRight', False)]
slow_down =[('KeyEvent', 'ArrowUp', False), ('KeyEvent', 'ArrowLeft', False), ('KeyEvent', 'ArrowRight', False),
            ('KeyEvent', 'ArrowDown', True)]
boost = [('KeyEvent', 'ArrowUp', True), ('KeyEvent', 'ArrowLeft', False), ('KeyEvent', 'ArrowRight', False),
         ('KeyEvent', 'x', True)]
#noop = [('KeyEvent', 'ArrowUp', False), ('KeyEvent', 'ArrowLeft', False), ('KeyEvent', 'ArrowRight', False)]


possible_actions = [left,right,forward,slow_down,boost]




while True:
  action_n = [np.random.choice(possible_actions) for ob in observation_n]  # your agent here
  observation_n, reward_n, done, info = env.step(action_n)
  env.render()

  if done[0]:
    print("finished round")
    print(reward_n[0])
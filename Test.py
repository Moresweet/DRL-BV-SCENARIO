import gymnasium as gym

env = gym.make('bv-scenario-v0', render_mode='rgb_array')
obs, info = env.reset()
env.render()
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
pass
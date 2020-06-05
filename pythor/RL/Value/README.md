# Value based algorithms

The main pytorch lightning training steps are written in `value_algos.py`

## Telegrad Support
Log your rewards in the `self.telegrad_logs` dictionary in the lightning Module in `value_algos.py`.

For example: `self.telegrad_logs={'rewards':total_reward, 'lr' self.lr, 'mean_rew':mean_rew}`

Here `mean_rew` is the mean of rewards for the last 100 episodes. 

If you want, you can change the status message you want over [here](https://github.com/nsidn98/PyThor/blob/776b3e7b006b9c8fa53d98388bccdb938e78645f/pythor/bots/rlCallback.py#L64) in `on_epoch_end` function. If you want to plot other parameters too change [here](https://github.com/nsidn98/PyThor/blob/776b3e7b006b9c8fa53d98388bccdb938e78645f/pythor/bots/rl_bot.py#L234) in `plot_diagrams` function.

## Usage
* Deep Q-Networks:
`python -W ignore pythor/RL/Value/value_algos.py --algo_name=dqn`

* Double Deep Q-Networks:
`python -W ignore pythor/RL/Value/value_algos.py --algo_name=ddqn`

* Dueling Double Deep Q-Networks:
`python -W ignore pythor/RL/Value/value_algos.py --algo_name=dddqn`

* Deep Q-Networks with priority experience buffer:
`python -W ignore pythor/RL/Value/value_algos.py --algo_name=dqn --priority=1`

Similarly you can run other algorithms in this folder with priority buffer.

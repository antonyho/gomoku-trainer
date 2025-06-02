# gomoku-trainer
AI Trainer of the classic eastern board game Gomoku


## Notice
<span style="color:red">_This is an experimental project_</span>


## About Gomoku
Gomoku, also known as Five in a Row, is a classic strategy board game where players alternate placing stones on a grid (typically 15x15 or 19x19) to form an unbroken line of five stones horizontally, vertically, or diagonally.


## Strategy

### Pre-trained Model
#### `stable-baselines3/ppo-CartPole-v1` [PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html) Algorithm[^1]
Since the game Gomoku requires sequential decision making, instead of understanding the context of a topic. A reinforcement learning model is more suitable. Pre-trained model like [`stable-baselines3/ppo-CartPole-v1`](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html) is a suitable option to be retrained for Gomoku.
#### [CNN](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html#stable_baselines3.ppo.CnnPolicy) Policy
CNN policy can be used for training the model with the 2D board game policy.

### Reinforcement Learning Gymnasium
#### [gym-gomoku](https://github.com/rockingdingo/gym-gomoku)
The gym-gomoku gymnasium can provide the OpenAI Gym environment for the PPO to perform reinforced training.

### Training Dataset
#### Self-play
The PPO can play against itself as training data using the PettingZoo gym.


## Usage
### Local Environment
#### Dependencies
[Dependenies for this project](./deps.md)

#### Build
`make`

#### Train
`make training`

### Docker Image
#### Build
```
docker build --tag gomoku-trainer .
```

#### Train
```
docker run gomoku-trainer:latest
```



[^1]: Proximal Policy Optimization Algorithms (https://arxiv.org/abs/1707.06347)


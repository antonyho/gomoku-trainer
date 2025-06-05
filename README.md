# gomoku-trainer

AI Trainer of the classic eastern board game Gomoku

## Notice

<span style="color:red">_This is an experimental project_</span>

## About Gomoku

Gomoku, also known as Five in a Row, is a classic strategy board game where players alternate placing stones on a grid (
typically 15x15 or 19x19) to form an unbroken line of five stones horizontally, vertically, or diagonally.

## Strategy

### Pre-trained Model

#### Train from scratch

Not using pre-trained model. Train with AlphaZero and MCTS algorithm.

#### [UCT](https://web.cs.du.edu/~sturtevant/papers/multi-player_UCT.pdf) Policy

UCT policy can be used for training the model which is employed in AlphaZero.

### Training Dataset

#### Self-play

Play against itself using the AlphaZero and TensorFlow.

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


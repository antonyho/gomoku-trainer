import pyspiel
from open_spiel.python import games  # pylint: disable=unused-import
from open_spiel.python.mfg import games as mfgs  # pylint: disable=unused-import
from open_spiel.python.algorithms.alpha_zero import alpha_zero

trained_model_filename = "gomoku_az_model"


def main():
    print("=" * 60)
    print("Gomoku Training with AlphaZero and MCTS algorithm")
    print("=" * 60)

    board_size = 15

    # Gomoku via MNK in OpenSpiel
    game = pyspiel.load_game("mnk", {'m': board_size, 'n': board_size, 'k': 5})
    config = alpha_zero.Config(  # Train with AlphaZero and TensorFlow
        game=game,
        path=trained_model_filename,
        learning_rate=0.001,
        weight_decay=1e-4,
        train_batch_size=128,
        replay_buffer_size=100000,
        replay_buffer_reuse=10,
        max_steps=1000000,
        checkpoint_freq=1000,

        # Self-play settings
        actors=8,  # Number of parallel self-play processes
        evaluators=2,
        evaluation_window=100,
        eval_levels=7,

        # MCTS settings
        uct_c=1.25,
        max_simulations=800,
        policy_alpha=0.3,
        policy_epsilon=0.25,
        temperature=1.0,
        temperature_drop=30,

        # Neural network architecture
        nn_model="resnet",
        nn_width=256,
        nn_depth=20,
        observation_shape=game.observation_tensor_shape(),
        output_size=game.num_distinct_actions(),

        quiet=False,
    )
    alpha_zero.alpha_zero(config)


if __name__ == "__main__":
    main()

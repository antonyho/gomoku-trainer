#!/usr/bin/env python3

"""
MCTS (Monte Carlo Tree Search) Bot for Gomoku
"""

import pyspiel
import numpy as np
from open_spiel.python.algorithms import mcts
import time

"""
1. NO Neural Network: The MCTS bot does NOT use any neural network
2. NOT Pre-trained: It's not trained at all - it calculates moves on-the-fly
3. Pure Search Algorithm: It explores possible future moves using random simulations

How MCTS Works:
1. Selection: Navigate the game tree using UCB (Upper Confidence Bound)
2. Expansion: Add a new node to the tree
3. Simulation: Play random moves until the game ends
4. Backpropagation: Update statistics back up the tree

The bot gets stronger with more simulations, but each move is calculated fresh.
"""


# ==================== Basic MCTS Bot Demo ====================

def create_mcts_bot_demo():
    print("=" * 60)
    print("MCTS Bot Demo - No Neural Network, Pure Search")
    print("=" * 60)

    # Create Gomoku game
    game = pyspiel.load_game("mnk(m=15,n=15,k=5)")  # Gomoku in m,n,k game rules

    rng = np.random.RandomState()
    rollout_count = 1
    evaluator = mcts.RandomRolloutEvaluator(rollout_count, rng)

    weak_simulation = 500
    intermediate_simulation = 2000
    strong_simulation = 10000

    # Create MCTS bots with different strengths
    weak_bot = mcts.MCTSBot(
        game,
        uct_c=2,  # Exploration constant
        max_simulations=weak_simulation,  # Very few simulations = weak play
        evaluator=evaluator,
        solve=True,  # Try to solve winning positions
        random_state=rng
    )

    medium_bot = mcts.MCTSBot(
        game,
        uct_c=2,
        max_simulations=intermediate_simulation,  # More simulations = stronger
        evaluator=evaluator,
        solve=True,
        random_state=rng
    )

    strong_bot = mcts.MCTSBot(
        game,
        uct_c=2,
        max_simulations=strong_simulation,  # Many simulations = much stronger
        evaluator=evaluator,
        solve=True,
        random_state=rng
    )

    print("\nCreated 3 MCTS bots:")
    print(f"- Weak bot: {weak_bot.max_simulations} simulations per move")
    print(f"- Medium bot: {medium_bot.max_simulations} simulations per move")
    print(f"- Strong bot: {strong_bot.max_simulations} simulations per move")
    print("\nNOTE: More simulations = more thinking time = stronger play")

    return game, weak_bot, medium_bot, strong_bot


# ==================== Demonstrate MCTS Strength ====================

def demonstrate_mcts_strength():
    """Show how simulation count affects playing strength"""

    game, weak_bot, medium_bot, strong_bot = create_mcts_bot_demo()

    print("\n" + "=" * 60)
    print("Demonstrating MCTS Playing Strength")
    print("=" * 60)

    # Test position: Create a position where there's a clear best move
    state = game.new_initial_state()

    # Set up a position where X has 3 in a row
    # X X X _ _
    # _ _ _ _ _
    # _ _ _ _ _
    moves = [0, 15, 1, 16, 2]  # X plays 0,1,2; O plays 15,16
    for move in moves:
        state.apply_action(move)

    print("\nTest Position (X to move):")
    print("X X X _ _ ...")
    print("O O _ _ _ ...")
    print("_ _ _ _ _ ...")
    print("\nBest move: Position 3 (completes 4 in a row)")

    # Get moves from each bot
    print("\nMCTS Bot Decisions:")

    # Weak bot
    start_time = time.time()
    weak_move = weak_bot.step(state)
    weak_time = time.time() - start_time
    print(f"Weak bot ({weak_bot.max_simulations} sims): Move {weak_move} - Time: {weak_time:.2f}s")

    # Medium bot
    start_time = time.time()
    medium_move = medium_bot.step(state)
    medium_time = time.time() - start_time
    print(f"Medium bot ({medium_bot.max_simulations} sims): Move {medium_move} - Time: {medium_time:.2f}s")

    # Strong bot
    start_time = time.time()
    strong_move = strong_bot.step(state)
    strong_time = time.time() - start_time
    print(f"Strong bot ({strong_bot.max_simulations} sims): Move {strong_move} - Time: {strong_time:.2f}s")

    print("\nObservation: Stronger bots (more simulations) are more likely")
    print("to find the best move, but take more time to think.")


# ==================== Interactive Play Against MCTS ====================

def play_against_mcts():
    """Play an interactive game against MCTS bot"""

    print("\n" + "=" * 60)
    print("Play Gomoku Against MCTS Bot")
    print("=" * 60)

    # Let player choose board size
    board_size = 15  # Standard Gomoku

    sim_counts = {"1": 1000, "2": 5000, "3": 10000, "4": 100000}

    # Let player choose difficulty
    print("\nChoose difficulty:")
    print(f"1. Easy ({sim_counts["1"]} simulations)")
    print(f"2. Medium ({sim_counts["2"]} simulations)")
    print(f"3. Hard ({sim_counts["3"]} simulations)")
    print(f"4. Expert ({sim_counts["4"]} simulations)")

    difficulty = input("Enter difficulty (1-4): ")
    simulations = sim_counts.get(difficulty, 500)

    # Create game and bot
    game = pyspiel.load_game(f"mnk(m={board_size},n={board_size},k=5)")
    rng = np.random.RandomState()
    rollout_count = 1
    evaluator = mcts.RandomRolloutEvaluator(rollout_count, rng)
    bot = mcts.MCTSBot(
        game,
        uct_c=2,
        max_simulations=simulations,
        solve=True,
        evaluator=evaluator,
        random_state=np.random.RandomState()
    )

    print(f"\nStarting game with {simulations} simulations per move")
    print("You are X, Bot is O")
    print("Enter moves as 'row col' (0-indexed, e.g., '7 7' for center)")

    state = game.new_initial_state()

    # Simple board display
    def display_board():
        print("\n  ", end="")
        for col in range(board_size):
            print(f"{col:2}", end=" ")
        print()

        for row in range(board_size):
            print(f"{row:2}", end=" ")
            for col in range(board_size):
                action = row * board_size + col
                if action in state.history():
                    idx = state.history().index(action)
                    print("X " if idx % 2 == 0 else "O ", end=" ")
                else:
                    print(". ", end=" ")
            print()

    # Game loop
    while not state.is_terminal():
        display_board()

        if state.current_player() == 0:  # Human turn
            while True:
                try:
                    move_str = input("\nYour move (row col): ")
                    if move_str.lower() == 'quit':
                        return
                    row, col = map(int, move_str.split())
                    action = row * board_size + col
                    if action in state.legal_actions():
                        state.apply_action(action)
                        break
                    else:
                        print("Invalid move! That position is taken or out of bounds.")
                except:
                    print("Invalid input! Use format: row col (e.g., '7 7')")
        else:  # Bot turn
            print("\nBot thinking...")
            start_time = time.time()
            action = bot.step(state)
            think_time = time.time() - start_time
            row, col = action // board_size, action % board_size
            print(f"Bot plays: {row} {col} (thought for {think_time:.1f}s)")
            state.apply_action(action)

    # Game over
    display_board()
    returns = state.returns()
    if returns[0] > 0:
        print("\nCongratulations! You win! 🎉")
    elif returns[1] > 0:
        print("\nBot wins! Better luck next time! 🤖")
    else:
        print("\nIt's a draw! Well played! 🤝")


# ==================== MCTS Analysis ====================

def analyze_mcts_decision_making():
    """Show how MCTS evaluates positions"""

    print("\n" + "=" * 60)
    print("MCTS Decision Making Process")
    print("=" * 60)

    # Create a custom MCTS bot that exposes statistics
    class AnalysisMCTSBot(mcts.MCTSBot):
        def step(self, state, return_stats=False):
            """Extended step function that can return statistics"""
            # Run MCTS
            # root = self._search(state)
            root = self.mcts_search(state)

            if return_stats:
                # Collect statistics about each move
                stats = {}
                for action in state.legal_actions():
                    if action in root.children:
                        child = root.children[action]
                        stats[action] = {
                            'visits': child.explore_count,
                            'wins': child.total_reward,
                            'win_rate': child.total_reward / child.explore_count if child.explore_count > 0 else 0
                        }
                    else:
                        stats[action] = {'visits': 0, 'wins': 0, 'win_rate': 0}

                # Get best action
                best_action = root.best_child()

                return best_action, stats
            else:
                return super().step(state)

    # Create game and analysis bot
    game = pyspiel.load_game("mnk(m=9,n=9,k=5)")  # Smaller for visibility
    bot = AnalysisMCTSBot(
        game,
        uct_c=2,
        max_simulations=1000,
        solve=True,
        evaluator=mcts.RandomRolloutEvaluator()
    )

    # Create a position
    state = game.new_initial_state()
    moves = [40, 41, 31, 32, 22]  # Some moves
    for move in moves:
        state.apply_action(move)

    print("\nAnalyzing position after 5 moves...")
    print("Running 1000 MCTS simulations...")

    # Get move with statistics
    action, stats = bot.step(state, return_stats=True)

    # Show top 5 moves by visit count
    print("\nTop 5 moves by MCTS visit count:")
    sorted_actions = sorted(stats.keys(), key=lambda a: stats[a]['visits'], reverse=True)[:5]

    for i, action in enumerate(sorted_actions):
        row, col = action // 9, action % 9
        s = stats[action]
        print(f"{i + 1}. Move ({row},{col}): {s['visits']} visits, "
              f"{s['win_rate']:.1%} win rate")

    row, col = action // 9, action % 9
    print(f"\nMCTS chooses: ({row},{col})")
    print("\nNote: MCTS explores promising moves more deeply.")


# ==================== Compare MCTS vs Random ====================

def mcts_vs_random():
    """Show MCTS bot playing against random player"""

    print("\n" + "=" * 60)
    print("MCTS Bot vs Random Player")
    print("=" * 60)

    game = pyspiel.load_game("mnk(m=9,n=9,k=5)")  # Smaller for speed
    evaluator = mcts.RandomRolloutEvaluator()
    mcts_bot = mcts.MCTSBot(
        game,
        uct_c=2,
        max_simulations=200,
        solve=True,
        evaluator=evaluator
    )

    wins = {'mcts': 0, 'random': 0, 'draw': 0}

    print("\nPlaying 10 games...")
    for game_num in range(10):
        state = game.new_initial_state()

        # Alternate who goes first
        mcts_first = game_num % 2 == 0

        while not state.is_terminal():
            if (state.current_player() == 0) == mcts_first:
                # MCTS move
                action = mcts_bot.step(state)
            else:
                # Random move
                action = np.random.choice(state.legal_actions())
            state.apply_action(action)

        # Check winner
        returns = state.returns()
        if mcts_first:
            if returns[0] > 0:
                wins['mcts'] += 1
            elif returns[1] > 0:
                wins['random'] += 1
            else:
                wins['draw'] += 1
        else:
            if returns[1] > 0:
                wins['mcts'] += 1
            elif returns[0] > 0:
                wins['random'] += 1
            else:
                wins['draw'] += 1

        print(f"Game {game_num + 1}: {'MCTS' if (returns[0] > 0) == mcts_first else 'Random'} wins")

    print(f"\nResults: MCTS: {wins['mcts']}, Random: {wins['random']}, Draw: {wins['draw']}")
    print("Observation: MCTS should win most games against random play!")


# ==================== Main Demo ====================

def main():
    print("MCTS Bot for Gomoku")
    print("=" * 60)

    while True:
        print("\nOptions:")
        print("1. Play against MCTS bot")
        print("2. See MCTS strength demonstration")
        print("3. Analyze MCTS decision making")
        print("4. Watch MCTS vs Random player")
        print("5. Exit")

        choice = input("\nChoose option (1-5): ")

        if choice == '1':
            play_against_mcts()
        elif choice == '2':
            demonstrate_mcts_strength()
        elif choice == '3':
            analyze_mcts_decision_making()
        elif choice == '4':
            mcts_vs_random()
        elif choice == '5':
            break
        else:
            print("Invalid choice!")


if __name__ == "__main__":
    main()
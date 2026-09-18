import numpy as np
from tictactoe import TicTacToeEnv
from network import Network  
import random

inputs = 9
hidden = [56]
outputs = 9

MIN_MAX = True     # True is MINMAX False is Random move opponent, 1000 games

AGENT_SIDE = 1  # 1 is first player -1 is second player



if MIN_MAX == False:   
    training = 1000
else:
    training = 1

# Initialize and load your master network champion
network = Network(inputs, hidden, outputs)
network.load("tictactoe_model.npz")

env = TicTacToeEnv()

wins = 0
losses = 0
draws = 0
forfeits = 0
opponent_forfeits = 0

for x in range(training):
    state, player, is_over = env.reset()


    while not is_over:

        if player == AGENT_SIDE:
            probs = network.forward(state)
            action = int(np.argmax(probs))

            state, player, reward, is_over = env.step(action)
            
            if is_over:
                if reward == 3:
                    wins += 1
                elif reward == -15:
                    forfeits += 1
                else:
                    draws += 1
                break
                   

        else:
            valid_actions = np.where(state[0:9] == -0.1)[0]
            if MIN_MAX:
                action = env.get_best_move()
            else:
                action = random.choice(valid_actions)
            
            state, last_player, reward, is_over = env.step(action)

            if is_over:
                if env.check_win(last_player):
                    losses += 1
                elif reward == -15:
                    opponent_forfeits += 1
                else:
                    draws += 1
                break

        player = env.player
    if x % 10 == 0:
        print(f"wins : {wins}, losses:{losses}, forfeits:{forfeits}, draws:{draws}, opponent_forfeits: {opponent_forfeits}")

print()
print("             ||         Final result       ||              ")
print(f"wins : {wins}, losses:{losses}, forfeits:{forfeits}, draws:{draws}, opponent_forfeits: {opponent_forfeits}")       

import numpy as np
import random
from tictactoe import TicTacToeEnv 
from network import Network
from logger import TrainingLog
from logger import Visualizer

inputs = 9 
hidden = [88]
outputs = 9

AGENT_SIDE = 1  # 1 to make it play as X (first player), -1 to make it play as O (second player)


def main():
    # Initializing
    agent = Network(inputs, hidden, outputs)
    env = TicTacToeEnv()
    logger = TrainingLog()
    visualizer = Visualizer()
    avg_loss = 0

    training = 6_000_000

    wins = 0
    draws = 0
    forfeits = 0
    losses = 0

    best_eval_score = -9999.0

    start_lr = 0.001
    end_lr = 0.0003
    decay_step = (start_lr - end_lr) / 500_000
    
    if AGENT_SIDE == 1:
        model_name = "tictactoe_model_x"
        training_log_name = "training_log_x"
    else:
        model_name = "tictactoe_model_o"
        training_log_name = "training_log_o"

    #start_eps = 1
    #end_eps = 0.05
    #decay_eps = eps_decay_step = (start_eps - end_eps) / 450_000 
    
    # Starts training
    for x in range(training):
        current_lr = max(end_lr, start_lr - (x * decay_step))
        #current_eps = max(end_eps, start_eps - (x * decay_eps))

        states_p1 = []
        probs_p1 = []
        ans_p1 = []
        t_rewards = []


        board, current_player, is_over = env.reset()

        while not is_over:                
            if current_player == AGENT_SIDE:
                '''if random.random < current_eps:
                    valid_actions = np.where((board[0:9] == -0.1) & (board[9:18] == -0.1))[0]
                    action = random.choice(valid_actions)
                else:'''             

                probability = agent.forward(board)
                action = np.argmax(probability)
                next_board, last_player, reward, is_over = env.step(action)
                
                states_p1.append(board.copy())
                ans_p1.append(action)
                probs_p1.append(probability)
                t_rewards.append(reward)

                if is_over:
                    if reward == 3:
                        wins += 1
                    elif reward == -15:
                        forfeits += 1
                    else:
                        draws += 1

                    
                    break
            else:
                valid_actions = np.where(board == -0.1)[0]
                chosen_action = None

                bot_focus_accuracy = min(0.8, 0.05 + (0.75 / 150_000) * x)    
                
                if random.random() < bot_focus_accuracy:
                    # Win Check: Can the opponent win right now?
                    for act in valid_actions:
                        env.board[act] = env.player
                        if env.check_win(env.player):
                            chosen_action = act
                        env.board[act] = 0
                        if chosen_action is not None:
                            break
                    
                    # Block Check: Is the agent about to win? Block them.
                    if chosen_action is None:
                        for act in valid_actions:
                            env.board[act] = -env.player
                            if env.check_win(-env.player):
                                chosen_action = act
                            env.board[act] = 0
                            if chosen_action is not None:
                                break

                if chosen_action is None:
                    action = random.choice(valid_actions)
                else:
                    action = chosen_action

                next_board, last_player, reward, is_over = env.step(action)


                if is_over:
                    if env.check_win(last_player):
                        losses += 1
                        if len(t_rewards) > 0:
                            t_rewards[-1] = -10  # -10 to discourge losing 
                    else:
                        draws += 1
                    break

            board = next_board
            current_player = env.player
        rewards = discount_rewards(t_rewards)
        avg_loss += agent.train(states_p1, ans_p1, probs_p1, rewards, current_lr)

        # Network pruning
        if x % 100_000 == 0 and x >= 100_000:
            agent.prune(0.1)

        # Evaluation
        # 1 games against min max, 10_000 games against randoms, optimal should be 0 forfeits and 0 losses
        if x % 50000 == 0 and x >= 200_000:
            print("Running Evaluation")
            eval_games = 1
            random_games = 20_000
            eval_wins = 0
            eval_draws = 0
            eval_forfeits = 0
            eval_losses = 0
            
            for i in range(eval_games):
                e_board, e_player, e_over = env.reset()
                
                while not e_over:
                    if e_player == AGENT_SIDE:
                        e_probs = agent.forward(e_board)
                        e_action = np.argmax(e_probs)
                        e_board, e_player, e_rew, e_over = env.step(e_action)
                        if e_over:
                            if e_rew == 3: eval_wins += 1
                            elif e_rew == -15: eval_forfeits += 1
                            else:
                                eval_draws += 1
                            break
                    else:
                        e_action = env.get_best_move()
                        e_board, last_p, e_rew, e_over = env.step(e_action)
                        
                        if e_over:
                            if env.check_win(last_p): 
                                eval_losses += 1
                            else: 
                                eval_draws += 1

                    e_player = env.player

            for i in range(random_games):
                e_board, e_player, e_over = env.reset()
                
                while not e_over:
                    if e_player == AGENT_SIDE:
                        e_probs = agent.forward(e_board)
                        e_action = np.argmax(e_probs)
                        e_board, _, e_rew, e_over = env.step(e_action)
                        if e_over:
                            if e_rew == 3: eval_wins += 1
                            elif e_rew == -15: eval_forfeits += 1
                            else:
                                eval_draws += 1
                            break
                    else:
                        valid_actions = np.where(e_board == -0.1)[0]
                        action = random.choice(valid_actions)
                        e_board, last_p, e_rew, e_over = env.step(action)

                        if e_over:
                            if e_rew == 3: eval_losses += 1
                            else: 
                                eval_draws += 1
                            break

                    e_player = env.player
        
            current_eval_score = (eval_wins * 2.0) + (eval_draws * 1.0) - (eval_losses * 2.0) - (eval_forfeits * 10.0)
            print(f"EVALUATION SCORE: {current_eval_score}, wins: {eval_wins}, losses: {eval_losses}, Draws: {eval_draws}, forfeits: {eval_forfeits}")
            
            if current_eval_score > best_eval_score:
                best_eval_score = current_eval_score
                agent.save(model_name)
                print(f"🏆 NEW CHAMPION MODEL SEIZED AT GAME {x}! (Wins: {eval_wins} | Draws: {eval_draws} | Losses: {eval_losses} | Forfeits: {eval_forfeits})")

        '''if x >= 200_000 and x % 20000 == 0:
            agent.weight_decay(0.95)'''

        if x > 0 and x % 5000 == 0:
            logger.record(x, avg_loss=avg_loss/5000)
            avg_loss = 0
            visualizer.update(logger.history)
            logger.save(training_log_name)
            total_recorded = wins + draws + forfeits + losses
            if total_recorded > 0:
                win_rate = (wins / total_recorded) * 100
                forfeit_rate = (forfeits / total_recorded) * 100
                print(f"📊 Match Progress Check [{x}/{training}] -> Win Rate: {win_rate:.1f}% | Forfeits: {forfeits} ({forfeit_rate:.1f}%) | Draws: {draws} | Losses: {losses}")

    print("Training done")


def discount_rewards(rewards_list, gamma=0.9):
    last = 0
    discounted = np.array(rewards_list, dtype=np.float32)
    for t in reversed(range(len(discounted))):
        discounted[t] = discounted[t] + last * gamma
        last = discounted[t]
    return discounted.tolist()

if __name__ == "__main__":
    main()

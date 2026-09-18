import numpy as np

class TicTacToeEnv:
    def __init__(self):
        self.board = np.zeros(9, dtype=np.int32) 
        self.over = False
        self.player = 1

    def reset(self):
        self.board = np.zeros(9, dtype=np.int32)
        self.over = False
        self.player = 1
        return self._get_encoded_state(), self.player, self.over

    def step(self, action):
        # 1. Check for illegal moves
        if self.board[action] != 0 or self.over:
            reward = -15
            self.over = True            
            return self._get_encoded_state(), self.player, reward, self.over

        # 2. Make the move
        self.board[action] = self.player

        # 3. Check if current player won
        if self.check_win(self.player):
            self.over = True
            reward = 3
            return self._get_encoded_state(), self.player, reward, self.over

        # 4. Check for a draw
        if 0 not in self.board:
            self.over = True
            reward = 0 
            return self._get_encoded_state(), self.player, reward, self.over

        # 5. Move is valid, game continues. Switch turns.
        reward = 0
        last_player = self.player
        self.player *= -1 
        
        return self._get_encoded_state(), last_player, reward, self.over

    def _get_encoded_state(self):
        # Maps state relative to the CURRENT player's turn
        my_map    = np.full(9, -0.1, dtype=np.float32)
        #enemy_map = np.full(9, -0.1, dtype=np.float32)

        for i in range(9):
            if self.board[i] == self.player:
                my_map[i] = 1.0
            elif self.board[i] == -self.player:
                my_map[i] = -1.0

        return np.concatenate([my_map])

    def check_win(self, p):
        b = self.board
        wins = [[0,1,2], [3,4,5], [6,7,8], [0,3,6], [1,4,7], [2,5,8], [0,4,8], [2,4,6]]
        return any(b[w[0]] == p and b[w[1]] == p and b[w[2]] == p for w in wins)


    def get_best_move(self):
        """Wrapper function to find the absolute best move for self.player."""
        best_score = -float('inf')
        best_action = None
        
        # Look at all possible moves on the current board
        for action in range(9):
            if self.board[action] == 0:
                # Simulate making the move
                self.board[action] = self.player
                
                # Check the score of this path (it's the opponent's turn next in simulation)
                score = self.min_max(is_maximizing=False, depth=0, original_player=self.player)
                
                # Undo the move
                self.board[action] = 0
                
                # We want the highest score possible for the original player
                if score > best_score:
                    best_score = score
                    best_action = action
                    
        return best_action

    def min_max(self, is_maximizing, depth, original_player):
        if self.check_win(original_player):
            return 10 - depth  # Win (subtract depth to prefer faster wins)

        if self.check_win(-original_player):
            return -10 + depth # Loss (add depth to prefer stalling a loss)

        if 0 not in self.board:
            return 0           # Draw

        # Figure out who is moving in this layer of simulation
        sim_player = original_player if is_maximizing else -original_player

        if is_maximizing:
            best_score = -float('inf')
            for action in range(9):
                if self.board[action] == 0:
                    self.board[action] = sim_player
                    score = self.min_max(False, depth + 1, original_player)
                    self.board[action] = 0
                    best_score = max(score, best_score)
            return best_score
        else:
            best_score = float('inf')
            for action in range(9):
                if self.board[action] == 0:
                    self.board[action] = sim_player
                    score = self.min_max(True, depth + 1, original_player)
                    self.board[action] = 0
                    best_score = min(score, best_score)
            return best_score

    def display_board(self):
        # Dictionary to map numbers to characters
        symbols = {1: 'X', -1: 'O', 0: '.'}
        
        for i in range(3):       
            for j in range(3):   
                index = i * 3 + j
                cell_value = self.board[index]
                
                print(symbols[cell_value], end=' ')
            print() 
        print()

if __name__ == "__main__":
    env = TicTacToeEnv()
    state, player, is_over = env.reset()
    print("Initial State shape:", state.shape)

    while not is_over:
        print(f"\nPlayer {player}'s turn.")
        try:
            move = int(input("Enter move (0-8): "))
            if move < 0 or move > 8:
                print("Invalid input! Put 0-8.")
                continue
        except ValueError:
            print("Please enter an integer.")
            continue

        state, player, reward, is_over = env.step(move)
        env.display_board()

        if is_over:
            print("Game over")
            break

        min_max_move = env.get_best_move()
        state, player, reward, is_over = env.step(min_max_move)
        env.display_board()
        
        if is_over:
            print("Game over")
            break

import numpy as np

class Network:
    """
    A fully-connected feedforward neural network built from scratch using
     NumPy. Supports arbitrary depth, leaky ReLU activations, softmax
    output, and REINFORCE-style policy gradient updates.
    """

    def __init__(self, input_size, hidden_layers, output_size):
        self.architecture = [input_size] + hidden_layers + [output_size]
        
        self.weights = []
        self.biases = []
        
        for i in range(len(self.architecture) - 1):
            inputs_to_layer = self.architecture[i]
            outputs_from_layer = self.architecture[i+1]
            
            w = np.random.uniform(-1, 1, (inputs_to_layer, outputs_from_layer))
            b = np.random.uniform(-1, 1, (1, outputs_from_layer))
            
            self.weights.append(w)
            self.biases.append(b)


    def forward(self, inputs):
        """
        Forward pass. Returns the softmax probability distribution over
        actions.
        """

        current_activation = np.array(inputs).reshape(1, -1)

        for i in range(len(self.weights) - 1):
            pre_activation = np.dot(current_activation, self.weights[i]) + self.biases[i]
            current_activation = np.where(pre_activation < 0, 0.1 * pre_activation, pre_activation)
            
        final_pre_activation = np.dot(current_activation, self.weights[-1]) + self.biases[-1]
        
        max_val = np.max(final_pre_activation, axis=1, keepdims=True)
        exp_probs = np.exp(final_pre_activation - max_val)
        probabilities = exp_probs / np.sum(exp_probs, axis=1, keepdims=True)
        probs_flat = probabilities.flatten()
        return probs_flat


    def train(self, inputs, action, probability, reward, learning_rate=0.1):
        """
        Policy gradient update (REINFORCE).

        inputs:     batch of states, shape (batch, input_size)
        action:     actions taken, shape (batch,)
        probability: action probabilities from forward(), shape (batch, output_size)
        reward:     discounted returns, shape (batch, 1)

        Returns the mean absolute error, useful as a training evaluation.
        """


        inputs = np.array(inputs).reshape(-1, self.architecture[0])
        batch_size = len(inputs)
        
        activations = [inputs]
        pre_activations = []
        
        current_activation = inputs
        for i in range(len(self.weights) - 1):
            pre_activation = np.dot(current_activation, self.weights[i]) + self.biases[i]
            pre_activations.append(pre_activation)

            current_activation = np.where(pre_activation < 0, 0.1 * pre_activation, pre_activation)
            activations.append(current_activation)
            
        final_pre_activation = np.dot(current_activation, self.weights[-1]) + self.biases[-1]
        pre_activations.append(final_pre_activation)
        
        action = np.array(action, dtype=int).flatten()
        probability = np.array(probability).reshape(batch_size, -1)
        reward = np.array(reward).reshape(-1, 1)

        move_one_hot = np.zeros_like(probability)
        move_one_hot[np.arange(batch_size), action] = 1.0

        error = (move_one_hot - probability) * reward
        
        current_gradient = error
        for i in reversed(range(len(self.weights))):
            layer_input = activations[i]

            weight_update = np.dot(layer_input.T, current_gradient) / batch_size
            bias_update = np.mean(current_gradient, axis=0, keepdims=True)

            self.weights[i] += weight_update * learning_rate
            self.biases[i]  += bias_update * learning_rate
            
            if i > 0:
                slope = np.where(pre_activations[i-1] < 0, 0.1, 1)
                current_gradient = np.dot(current_gradient, self.weights[i].T) * slope

        return float(np.mean(np.abs(error)))

    def prune(self, threshold):
        zeros = 0
        total_weights = 0
        
        for i in range(len(self.weights)):
            self.weights[i] = np.where(np.abs(self.weights[i]) < threshold, 0.0, self.weights[i])
    
            zeros += np.sum(self.weights[i] == 0.0)
            total_weights += self.weights[i].size

        print(f"Pruning complete: {zeros} zeroed out of {total_weights} total weights.")
        return zeros, total_weights # Optional


    def weight_decay(self, decay = 0.9):
        for i in range(len(self.weights)):
                self.weights[i] *= decay


    def save(self, file_path="model.npz"):
        """ Saves the current MODEL for later use or more training"""

        weights_obj = np.empty(len(self.weights), dtype=object)
        for idx, w in enumerate(self.weights):
            weights_obj[idx] = w

        biases_obj = np.empty(len(self.biases), dtype=object)
        for idx, b in enumerate(self.biases):
            biases_obj[idx] = b

        np.savez_compressed(
            file_path,
            architecture=np.array(self.architecture),
            weights=weights_obj,
            biases=biases_obj,
        )
        print("Saved")
        
        
    def load(self, file_path="model.npz"):
        """ Loads up a model from the file_path"""

        data = np.load(file_path, allow_pickle=True)
        # allow_pickle=True is required because weights are stored as an
        # object array of variable-shaped numpy arrays, not a single tensor.
        self.architecture = list(data['architecture'])
        self.weights = list(data['weights'])
        self.biases = list(data['biases'])
        print(f"{file_path} loaded")


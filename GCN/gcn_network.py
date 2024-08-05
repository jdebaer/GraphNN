import numpy as np
from gcn_layer import GCNLayer
from softmax_layer import SoftmaxLayer


class GCNNetwork():
    
    def __init__(self, n_inputs, n_outputs, hidden_sizes, seed=0):

        np.random.seed(seed)

        self.layers =  []

        # Input layer
        input_layer = GCNLayer(n_inputs, hidden_sizes[0])
        self.layers.append(input_layer)

        print(f"Input layer: ({n_inputs}, {hidden_sizes[0]})")

        # Hidden layers
        for idx in range(len(hidden_sizes)):
            gcn_layer = GCNLayer(hidden_sizes[idx], hidden_sizes[idx+1])
            self.layers.append(gcn_layer)

            print(f"Hidden layer: ({hidden_sizes[idx]}, {hidden_sizes[idx+1]})")

            if (idx + 1) > (len(hidden_sizes) - 2):
                break

        # Softmax layer
        softmax_layer = SoftmaxLayer(hidden_sizes[-1], n_outputs)
        self.layers.append(softmax_layer)

        print(f"Softmax layer: ({hidden_sizes[-1]}, {n_outputs})")
    
    def __repr__(self):
        return '\n'.join([str(layer) for layer in self.layers])

    # Forward pass
    def forward(self, A, H):

        for layer in self.layers[:-1]:
            H = layer.forward(A, H)

        probabilities = self.layers[-1].forward(H)
                
        #return np.asarray(probabilities) # Should already be np array
        return probabilities



def main():
    gcn_network = GCNNetwork(
				n_inputs = 32,
				n_outputs = 3,
				hidden_sizes = [16, 2],
				seed = 100,
			    )
    print(gcn_network)
				

if __name__ == "__main__":
    main()
            




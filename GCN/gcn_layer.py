from helpers import glorot_init
import numpy as np

class GCNLayer():
    def __init__(self, n_inputs, n_outputs):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

        # We need to transpose W because we put W in front of A_hat.
        self.W = glorot_init(self.n_outputs, self.n_inputs)
        self.H_next_still_transposed = None # Used to calculate gradients
        self.A_hat = None

    def __repr__(self):
        return f"Conv. layer: W is ({self.n_inputs}, {self.n_outputs})"

    def forward(self, A_hat, H): # A_hat is normalized adj. matrix
        
        # Essentially we have W @ A_hat @ H
        # dim of H is (batch_size, n_inputs)
        # dim of A_hat is (batch_size, batch_size)
        # dim of W is (n_outputs, n_inputs), special for GCN 
        # A_hat @ H -> (batch_size, n_inputs) so (1,3,4) in our example

        self.A_hat = A_hat #(batch_size, batch_size)

        self.transposed = (A_hat @ H).T  # (n_inputs, batch_size) for (4,3,1) in our example

        # self.W is (n_outputs, n_inputs) # so (3,4) in our example
        # Note that it's one and the same self.W which we apply to all features of all nodes in all batches

        #print(self.W.shape) -> (16, 32)
        #print(self.transposed.shape) -> (34, 34)

        H_lin_transp = self.W @ self.transposed # (n_outputs, n_inputs) @ (n_inputs, b_s) -> (n_outputs, b_s) 
        # We have all the first features in a row in the first dim
  
        # tanh simply scales every number to (-1,1)
        self.H_next_still_transposed = np.tanh(H_lin_transp)

        return self.H_next_still_transposed.T  # T brings it back to (batch_size, n_outputs) 

    def backward(self, optim, update=True):

        # H_next_still_transposed are the non-linear outputs, so this take the place of tanh("linear outputs")

        dtanh = 1 - np.asarray(self.H_next_still_transposed.T)**2         # (batch_size, n_outputs)

        der_tanh_wrt_lin_out = np.multiply(optim.out,dtanh)     # (batch_size, n_outputs) = (batch_size, n_outputs) * (batch_size, n_outputs) 

        # optim.out == gradient for the nonlinear outputs (above activation function)
        # The gradient of the linear output (below) is obtained by element multiplying the gradient of the nonlinear output (which is 
        # coming in via backpropagation, via optim.out here) with the result of pushing the nonlinear output through the derivative
        # of the activation function.   

        # (b_s, n_inputs) = (b_s, n_outputs) @ (n_outputs, n_inputs)
        optim.out = der_tanh_wrt_lin_out @ self.W	

        # This is part. der. wrt A_hat @ H, but we need pd wrt H, so we need to do an additional @ with A_hat 
        # which essentially means we're calculating another partial derivative

        # (b_s, b_s) @ (b_s, n_inputs) -> (b_s, n_inputs)
        optim.out = self.A_hat @ optim.out

        # (n_outputs, b_s) @ (b_s, n_inputs) -> (n_outputs, n_inputs)
        # Here the x1,x2,... are the result of A_hat @ H which we saved as self.transposed
        gradient_W = np.asarray(der_tanh_wrt_lin_out.T @ self.transposed.T) 

        gradient_W = gradient_W / optim.bs

        gradient_W_wd = self.W * optim.wd / optim.bs

        if update:
            self.W -= (gradient_W + gradient_W_wd) * optim.lr

        return gradient_W, gradient_W_wd
        
def main():
    gcn_layer = GCNLayer(4,3)
    print(gcn_layer)

    # (batch_size, batch_size) here (3, 3)
    A_hat = np.array([
         [1,1,1],
         [1,1,1],
         [1,1,1]
        ])
    # (batch_size, n_features) here (3, 4)
    H = np.array([
         [1,2,3,4],     # Sample 1 == node 1 w/ 4 features
         [5,6,7,8],     # Sample 2 == node 2 w/ 4 features
         [9,10,11,12]   # Sample 3 == node 3 w/ 4 features
        ])
    gcn_layer.forward(A_hat, H)

if __name__ == "__main__":
    main()

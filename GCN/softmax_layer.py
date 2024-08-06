from helpers import glorot_init
import numpy as np

class SoftmaxLayer():

    def __init__(self, n_inputs, n_outputs):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.W = glorot_init(self.n_outputs, self.n_inputs)
        self.b = np.zeros(shape=(self.n_outputs,1))
        self.transposed = None # Used to calculate gradients

    def __repr__(self):
        return f"Softmax: W is ({self.n_inputs}, {self.n_outputs})"

    def forward(self, H):
    
        # H is (batch_size, n_features)

        self.transposed = H.T # Now (n_features (input), batch_size)

        #logits = np.asarray(self.W @ self.transposed) + b

        #print(self.W.shape) -> (3, 2)
        #print(self.transposed.shape) -> (16, 34)

        logits = (self.W @ self.transposed) + self.b

        # Subtracting the max supposedly adds numerical stability (look into this)
        logits_min_max = logits - np.max(logits, axis=0, keepdims=True)
        exps = np.exp(logits_min_max)
        probs =  exps / np.sum(exps, axis=0, keepdims=True)

        # Transpose back and return
        return probs.T

    def backward(self, optim, update=True):

        # In essence here we:
        #    1. Calculate our gradient
        #    2. Use it to update our weights
        #    3. Store the gradient in the optimizer's _out for the next layer

        train_mask = np.zeros(optim.y_pred.shape[0]) # batch size?
        train_mask[optim.train_nodes] = 1
        # This is equiv to an unsqueeze: np.expand_dims(test, axis=-1)
        train_mask = train_mask.reshape((-1,1))

        #loss = np.asarray((optim.y_pred - optim.y_true))
        # This is d of CELoss wrt Softmax's inputs (logits): 
        #https://math.stackexchange.com/questions/945871/derivative-of-softmax-loss-function
        #https://davidbieber.com/snippets/2020-12-12-derivative-of-softmax-and-the-softmax-cross-entropy-loss/

        deriv_cel_wrt_sm = optim.y_pred - optim.y_true

        # (batch_size, n_outputs)
        deriv_cel_wrt_sm = np.multiply(deriv_cel_wrt_sm, train_mask) # Set pd to 0 for unused nodes
        
        # Project the derivatives "out" to the lower layer, store in optim
        # What we're doing here is redistributing the gradients to the lower nodes, so that the in the lower layer we can
        # come in with a gradient for each node.
        # (batch_size, n_outputs)*(n_outputs,n_inputs) -> (batch_size,n_inputs)
        # w/ n_inputs defined in the lower layer.
        optim.out = deriv_cel_wrt_sm @ self.W 

        # (n_outputs,n_inputs) = (n_outputs,batch_size)*(batch_size,n_inputs)
        gradient_W = deriv_cel_wrt_sm.T @ self.transposed.T
       
        # The above works as follows: left side contains d cel / d sm, right side contains the x1,x2,x3 etc. used to multiply with the w's.
        # To get the partial derivate d sm / d w<index>, we basically only need to keep the corresponding x<index> since all other w's are "constants".
        # Example: let's say outer is 2 and inner is 3 and batch size is 1, so we have 2x1 @ 1x3 -> 2x3
        # 
        # d1 @ x1 x2 x2 -> d1x1 d1x2 d1x3 which are the partial derivatives for w11 w12 w13
        # d2               d2x1 d2x2 d2x3                                       w21 w22 w23
        #
	# These will be added up for each cell across the batch, so we need to average out:

        gradient_W = gradient_W / optim.bs

        # Sum over batches, div by n of batches
        gradient_b = deriv_cel_wrt_sm.T.sum(axis=1, keepdims=True)
        gradient_b = gradient_b / optim.bs

        # Weight decay is a regularization technique that adds a penalty to the 
        # loss function, usually the L2 norm of the weights, to prevent overfitting 
        # and improve generalization in neural networks. By default, PyTorch applies 
        # weight decay to both weights and biases simultaneously, but it can be 
        # configured to handle different parameters. 
        # However, some people prefer to only apply weight decay to the weights.

        gradient_W_wd = self.W * optim.wd / optim.bs

        if update:
            self.W -= (gradient_W + gradient_W_wd) * optim.lr
            self.b -= gradient_b.reshape(self.b.shape) * optim.lr

        return gradient_W + gradient_W_wd, gradient_b.reshape(self.b.shape)

def main():
    
    softmax_layer = SoftmaxLayer(4,4)
    print(softmax_layer)

if __name__ == "__main__":
    main()

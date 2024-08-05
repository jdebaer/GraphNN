
class GDOptim():

    # In PyTorch, backward() has each tensor in the model calculate its gradient (stored in tensor)
    # optim.step() then walks over the tensors and has each tensor's ".grad" to update the tensor
    # values (weights) 

    # The roll of this object is to flow the gradients down the layers during backprop.

    # Each layer:
    #        1. Calculates its gradient (starting with softmax and then down the GCN layers)
    #        2. Updates its parameters (W)
    #        3. Stores the gradients that need to flow into the next layer in self._out


    def __init__(self,lr,wd):
        
        self.lr = lr
        self.wd = wd

        self.y_pred = None
        self.y_true = None
        self._out = None
        self.bs = None
        self.train_nodes = None

    def __call__(self, y_pred, y_true, train_nodes):
    
        self.y_pred = y_pred
        self.y_true = y_true
        self.train_nodes = train_nodes
        self.bs = self.train_nodes.shape[0]

    @property
    def out(self):
        return self._out

    @out.setter
    def out(self, y):
        self._out = y

        


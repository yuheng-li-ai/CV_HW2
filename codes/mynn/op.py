from abc import abstractmethod
import numpy as np

class Layer():
    def __init__(self) -> None:
        self.optimizable = True
    
    @abstractmethod
    def forward():
        pass

    @abstractmethod
    def backward():
        pass


class Linear(Layer):
    """
    The linear layer for a neural network. You need to implement the forward function and the backward function.
    """
    def __init__(self, in_dim, out_dim, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.W = initialize_method(size=(in_dim, out_dim))
        self.b = initialize_method(size=(1, out_dim))
        self.grads = {'W' : None, 'b' : None}
        self.input = None # Record the input for backward process.

        self.params = {'W' : self.W, 'b' : self.b}

        self.weight_decay = weight_decay # whether using weight decay
        self.weight_decay_lambda = weight_decay_lambda # control the intensity of weight decay
            
    
    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def forward(self, X):
        """
        input: [batch_size, in_dim]
        out: [batch_size, out_dim]
        """
        self.input = X
        return X @ self.params['W'] + self.params['b']

    def backward(self, grad : np.ndarray):
        """
        input: [batch_size, out_dim] the grad passed by the next layer.
        output: [batch_size, in_dim] the grad to be passed to the previous layer.
        This function also calculates the grads for W and b.
        """
        batch_size = grad.shape[0]
        self.grads['W'] = self.input.T @ grad / batch_size
        self.grads['b'] = np.sum(grad, axis=0, keepdims=True) / batch_size
        return grad @ self.params['W'].T
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}

class conv2D(Layer):
    """
    The 2D convolutional layer. Try to implement it on your own.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.W = initialize_method(size=(1, out_channels, in_channels, kernel_size, kernel_size))
        self.b = initialize_method(size=(1, out_channels, 1, 1))
        self.params = {'W': self.W, 'b': self.b}
        self.grads = {'W': None, 'b': None}

        self.weight_decay = weight_decay
        self.weight_decay_lambda = weight_decay_lambda
        self.input = None
        self.padded_input = None
        self.input_windows = None

    def _extract_windows(self, X):
        windows = np.lib.stride_tricks.sliding_window_view(
            X,
            (self.kernel_size, self.kernel_size),
            axis=(2, 3)
        )
        return windows[:, :, ::self.stride, ::self.stride, :, :]

    def __call__(self, X) -> np.ndarray:
        return self.forward(X)
    
    def forward(self, X):
        """
        input X: [batch, channels, H, W]
        W : [1, out, in, k, k]
        no padding
        """
        self.input = X
        if self.padding > 0:
            self.padded_input = np.pad(
                X,
                ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                mode='constant'
            )
        else:
            self.padded_input = X

        batch_size, _, height, width = self.padded_input.shape
        out_height = (height - self.kernel_size) // self.stride + 1
        out_width = (width - self.kernel_size) // self.stride + 1

        self.input_windows = self._extract_windows(self.padded_input)
        output = np.tensordot(
            self.input_windows,
            self.params['W'][0],
            axes=([1, 4, 5], [1, 2, 3])
        )
        output = np.moveaxis(output, -1, 1)
        output += self.params['b']
        return output

    def backward(self, grads):
        """
        grads : [batch_size, out_channel, new_H, new_W]
        """
        batch_size, _, out_height, out_width = grads.shape
        grad_input_padded = np.zeros_like(self.padded_input)
        grad_b = np.sum(grads, axis=(0, 2, 3), keepdims=True).reshape(1, self.out_channels, 1, 1)
        grad_W = np.tensordot(
            grads,
            self.input_windows,
            axes=([0, 2, 3], [0, 2, 3])
        ).reshape(1, self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)

        for kh in range(self.kernel_size):
            h_slice = slice(kh, kh + self.stride * out_height, self.stride)
            for kw in range(self.kernel_size):
                w_slice = slice(kw, kw + self.stride * out_width, self.stride)
                grad_input_padded[:, :, h_slice, w_slice] += np.tensordot(
                    grads,
                    self.params['W'][0, :, :, kh, kw],
                    axes=([1], [0])
                ).transpose(0, 3, 1, 2)

        self.grads['W'] = grad_W / batch_size
        self.grads['b'] = grad_b / batch_size

        if self.padding > 0:
            return grad_input_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
        return grad_input_padded
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}
        
class ReLU(Layer):
    """
    An activation layer.
    """
    def __init__(self) -> None:
        super().__init__()
        self.input = None

        self.optimizable =False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        output = np.where(X<0, 0, X)
        return output
    
    def backward(self, grads):
        assert self.input.shape == grads.shape
        output = np.where(self.input < 0, 0, grads)
        return output

class MultiCrossEntropyLoss(Layer):
    """
    A multi-cross-entropy loss layer, with Softmax layer in it, which could be cancelled by method cancel_softmax
    """
    def __init__(self, model = None, max_classes = 10) -> None:
        super().__init__()
        self.model = model
        self.max_classes = max_classes
        self.has_softmax = True
        self.predicts = None
        self.labels = None
        self.probs = None
        self.grads = None
        self.optimizable = False

    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)
    
    def forward(self, predicts, labels):
        """
        predicts: [batch_size, D]
        labels : [batch_size, ]
        This function generates the loss.
        """
        self.predicts = predicts
        self.labels = labels
        if self.has_softmax:
            self.probs = softmax(predicts)
        else:
            self.probs = predicts

        eps = 1e-12
        batch_indices = np.arange(labels.shape[0])
        correct_class_prob = self.probs[batch_indices, labels]
        loss = -np.mean(np.log(correct_class_prob + eps))
        return loss
    
    def backward(self):
        # first compute the grads from the loss to the input
        batch_size = self.labels.shape[0]
        self.grads = self.probs.copy()
        self.grads[np.arange(batch_size), self.labels] -= 1
        self.grads /= batch_size
        # Then send the grads to model for back propagation
        self.model.backward(self.grads)

    def cancel_soft_max(self):
        self.has_softmax = False
        return self
    
class L2Regularization(Layer):
    """
    L2 Reg can act as weight decay that can be implemented in class Linear.
    """
    pass
       
def softmax(X):
    x_max = np.max(X, axis=1, keepdims=True)
    x_exp = np.exp(X - x_max)
    partition = np.sum(x_exp, axis=1, keepdims=True)
    return x_exp / partition

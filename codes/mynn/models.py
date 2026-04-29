from .op import *
import pickle

class Model_MLP(Layer):
    """
    A model with linear layers. We provied you with this example about a structure of a model.
    """
    def __init__(self, size_list=None, act_func=None, lambda_list=None):
        self.size_list = size_list
        self.act_func = act_func

        if size_list is not None and act_func is not None:
            self.layers = []
            for i in range(len(size_list) - 1):
                layer = Linear(in_dim=size_list[i], out_dim=size_list[i + 1])
                if lambda_list is not None:
                    layer.weight_decay = True
                    layer.weight_decay_lambda = lambda_list[i]
                if act_func == 'Logistic':
                    raise NotImplementedError
                elif act_func == 'ReLU':
                    layer_f = ReLU()
                self.layers.append(layer)
                if i < len(size_list) - 2:
                    self.layers.append(layer_f)

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        assert self.size_list is not None and self.act_func is not None, 'Model has not initialized yet. Use model.load_model to load a model or create a new model with size_list and act_func offered.'
        outputs = X
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads

    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            param_list = pickle.load(f)
        self.size_list = param_list[0]
        self.act_func = param_list[1]

        self.layers = []
        for i in range(len(self.size_list) - 1):
            layer = Linear(in_dim=self.size_list[i], out_dim=self.size_list[i + 1])
            layer.W = param_list[i + 2]['W']
            layer.b = param_list[i + 2]['b']
            layer.params['W'] = layer.W
            layer.params['b'] = layer.b
            layer.weight_decay = param_list[i + 2]['weight_decay']
            layer.weight_decay_lambda = param_list[i+2]['lambda']
            if self.act_func == 'Logistic':
                raise NotImplemented
            elif self.act_func == 'ReLU':
                layer_f = ReLU()
            self.layers.append(layer)
            if i < len(self.size_list) - 2:
                self.layers.append(layer_f)
        
    def save_model(self, save_path):
        param_list = [self.size_list, self.act_func]
        for layer in self.layers:
            if layer.optimizable:
                param_list.append({'W' : layer.params['W'], 'b' : layer.params['b'], 'weight_decay' : layer.weight_decay, 'lambda' : layer.weight_decay_lambda})
        
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)
        

class Model_CNN(Layer):
    """
    A model with conv2D layers. Implement it using the operators you have written in op.py
    """
    def __init__(self, input_shape=(1, 28, 28), num_classes=10, conv_channels=8, kernel_size=3, hidden_dim=128):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.conv_channels = conv_channels
        self.kernel_size = kernel_size
        self.hidden_dim = hidden_dim

        conv_out_h = input_shape[1] - kernel_size + 1
        conv_out_w = input_shape[2] - kernel_size + 1
        flatten_dim = conv_channels * conv_out_h * conv_out_w
        conv_scale = (2.0 / (input_shape[0] * kernel_size * kernel_size)) ** 0.5
        fc_scale = (2.0 / flatten_dim) ** 0.5

        self.layers = [
            conv2D(input_shape[0], conv_channels, kernel_size, initialize_method=lambda size: np.random.normal(0, conv_scale, size)),
            ReLU(),
            Linear(flatten_dim, hidden_dim, initialize_method=lambda size: np.random.normal(0, fc_scale, size)),
            ReLU(),
            Linear(hidden_dim, num_classes, initialize_method=lambda size: np.random.normal(0, (2.0 / hidden_dim) ** 0.5, size)),
        ]
        for layer in self.layers:
            if layer.optimizable:
                layer.params['b'] = np.zeros_like(layer.params['b'])
                layer.b = layer.params['b']
        self._cached_feature_shape = None

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        outputs = X
        for idx, layer in enumerate(self.layers):
            if idx == 2:
                self._cached_feature_shape = outputs.shape
                outputs = outputs.reshape(outputs.shape[0], -1)
            outputs = layer(outputs)
        return outputs

    def backward(self, loss_grad):
        grads = loss_grad
        for idx in range(len(self.layers) - 1, -1, -1):
            layer = self.layers[idx]
            grads = layer.backward(grads)
            if idx == 2:
                grads = grads.reshape(self._cached_feature_shape)
        return grads
    
    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            params = pickle.load(f)

        meta = params[0]
        self.__init__(
            input_shape=tuple(meta['input_shape']),
            num_classes=meta['num_classes'],
            conv_channels=meta['conv_channels'],
            kernel_size=meta['kernel_size'],
            hidden_dim=meta['hidden_dim'],
        )

        param_idx = 1
        for layer in self.layers:
            if layer.optimizable:
                layer.W = params[param_idx]['W']
                layer.b = params[param_idx]['b']
                layer.params['W'] = layer.W
                layer.params['b'] = layer.b
                layer.weight_decay = params[param_idx]['weight_decay']
                layer.weight_decay_lambda = params[param_idx]['lambda']
                param_idx += 1
        
    def save_model(self, save_path):
        param_list = [{
            'input_shape': self.input_shape,
            'num_classes': self.num_classes,
            'conv_channels': self.conv_channels,
            'kernel_size': self.kernel_size,
            'hidden_dim': self.hidden_dim,
        }]
        for layer in self.layers:
            if layer.optimizable:
                param_list.append({
                    'W': layer.params['W'],
                    'b': layer.params['b'],
                    'weight_decay': layer.weight_decay,
                    'lambda': layer.weight_decay_lambda,
                })

        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)

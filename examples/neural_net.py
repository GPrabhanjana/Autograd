from engine import Value
import numpy as np

class Neuron:
    def __init__(self, nin):
        self.w=[Value(np.random.randn()) for _ in range(nin)]
        self.b=Value(np.random.randn())
        
    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        out = act.tanh()
        return out
    
    def parameters(self):
        return self.w + [self.b]
    
class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]
        
    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs
    
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]
    
class MLP:
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]
        
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


n = MLP(2, [4, 4, 1])
xs = [[1, 1],
      [1, 0],
      [0, 1],
      [0, 0]]
ys = [1, 0, 0, 1]

for k in range(200):
    ypred = [n(x) for x in xs]
    loss = sum((yout[0] - ygt)**2 for ygt, yout in zip(ys, ypred))
    
    for p in n.parameters():
        p.grad = 0
        
    loss.backward()
    
    for p in n.parameters():
        p.data += -0.05 * p.grad
    print(k, loss.data)

print(ypred)

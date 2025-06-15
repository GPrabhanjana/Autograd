import math

class Value:
    
    def __init__(self, data, _children = ()):
        self.data = data
        self._prev = set(_children)
        self._backward = lambda: None
        self.grad = 0.0
        
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other))
        
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        
        out._backward = _backward
        
        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other))
        
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        
        out._backward = _backward
        
        return out
    
    def __pow__(self, other):
        if isinstance(other, (int, float)):
            out = Value(self.data**other, (self,))

            def _backward():
                self.grad += (other * self.data**(other-1)) * out.grad
            out._backward = _backward
            
            return out
        
        elif isinstance(other, Value):
            if self.data <= 0:
                raise ValueError("Base must be positive for Value^Value operations")
            
            # Compute a^b as exp(b * ln(a))
            ln_self = self.log()
            exponent_term = other * ln_self
            out = exponent_term.exp()
            
            return out
    
    def log(self):
        if self.data <= 0:
            raise ValueError("Cannot take log of non-positive number")
        
        out = Value(math.log(self.data), (self,))
        def _backward():
            self.grad += (1.0 / self.data) * out.grad
        out._backward = _backward
        return out
    
    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        out = Value(t, (self, ))
        
        def _backward():
            self.grad += (1 - t**2) * out.grad
        
        out._backward = _backward
        
        return out
    
    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,))

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out
    
    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self, ))
        
        def _backward():
            self.grad += out.data * out.grad
        
        out._backward = _backward
        
        return out
    
    def sin(self):
        out = Value(math.sin(self.data), (self,))
        def _backward():
            self.grad += math.cos(self.data) * out.grad
        out._backward = _backward
        return out

    def cos(self):
        out = Value(math.cos(self.data), (self,))
        def _backward():
            self.grad += -math.sin(self.data) * out.grad
        out._backward = _backward
        return out

    def tan(self):
        out = Value(math.tan(self.data), (self,))
        def _backward():
            self.grad += (1 / math.cos(self.data)**2) * out.grad
        out._backward = _backward
        return out
    
    def __rmul__(self, other):
        return self * other
    
    def __radd__(self, other):
        return self + other
    
    def __neg__(self):
        return self * -1
    
    def __sub__(self, other):
        return self + (-other)
    
    def __rsub__(self, other):
        return other + (-self)
    
    def __truediv__(self, other):
        return self * other**-1
    
    def __rtruediv__(self, other):
        return other * self**-1
    
    def backward(self):
        
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
        
    def __repr__(self):
        return f"Val(data={self.data})"
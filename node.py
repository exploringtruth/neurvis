import math
import numpy as np
import matplotlib.pyplot as plt
from utils import FormattedList

class Node:
    NODE_PARAMETERS = [
    "Data (data)",
    "Children (children)",
    "Operator (operator)",
    "Update (update)"
    ]

    BINARY_OPERATORS = [
        "Add {+}",
        "Subtract {-}",
        "Multiply {*}",
        "Divide {/}",
        "Power {^}"
    ]
    
    UNARY_OPERATORS = [
        "Absolute Value {.abs()}",
        "Exponential {.e()}",
        "Negate {-(x)}"
    ]
    
    ACTIVATION_FUNCTIONS = [
        "Tanh {tanh()}",
        "ReLU {relu()}",
        "Threshold ReLU {threshold_relu()}",
        "PReLU {prelu()}",
        "Leaky ReLU {leaky_relu()}",
        "Sigmoid {sigmoid()}",
        "SiLU {silu()}",
        "ELU {elu()}",
        "GELU {gelu()}",
        "SELU {selu()}",
        "Softplus {softplus()}",
        "Mish {mish()}",
        "Hardswish {hardswish()}"
    ]

    @classmethod
    def _format(cls, mapping):
        result = []
        for name, items in mapping:
            result.append(f"{name}:")
            result.extend(f"  - {item}" for item in items)
        return FormattedList(result)

    @classmethod
    def get_parameters(cls):
        return cls._format([("Weight Initialization Functions", cls.INSTANCE_PARAMETERS)])

    get_parameter = get_instance_parameter = get_instance_parameter = get_parameters

    @classmethod
    def get_all(cls):
        return cls._format([
            ("Binary Operators", cls.BINARY_OPERATORS),
            ("Unary Operators", cls.UNARY_OPERATORS),
            ("Element-Wise Activation Functions", cls.ACTIVATION_FUNCTIONS)
        ])

    @classmethod
    def get_operators(cls):
        return cls._format([
            ("Binary Operators", cls.BINARY_OPERATORS),
            ("Unary Operators", cls.UNARY_OPERATORS)
        ])

    @classmethod
    def get_binary_operators(cls):
        return cls._format([("Binary Operators", cls.BINARY_OPERATORS)])

    get_binary = get_binary_operators

    @classmethod
    def get_unary_operators(cls):
        return cls._format([("Unary Operators", cls.UNARY_OPERATORS)])

    get_unary = get_unary_operators

    @classmethod
    def get_activations(cls):
        return cls._format([("Element-Wise Activation Functions", cls.ACTIVATION_FUNCTIONS)])

    get_activation = get_activation_function = get_activation_functions = get_activations

    def __init__(self,data, children=(), operator='', update=True):
        self.data = data
        self.gradient = 0
        self.children = set(children) # Remove duplicate from tuple by turning it into set, turn set into list to access children through index
        self.operator = operator
        self._backward = lambda: None
        self.update = update

    def __repr__(self):
        return f"Node(data={self.data})"
        
    # Binary Operators

    def __add__(self, other):
        other = other if isinstance(other, Node) else Node(other) # isinstance(object, class) checks if 'object' is a 'class' 
        output = Node(self.data + other.data, children=(self,other), operator='+')

        def add_grad():
            self.gradient += output.gradient
            other.gradient += output.gradient

        output._backward = add_grad # Stores function inside an instance variable
        return output

    def __sub__(self, other):
        other = other if isinstance(other, Node) else Node(other)
        output = Node(self.data - other.data, children=(self,other), operator='-')

        def sub_grad():
            self.gradient += output.gradient
            other.gradient -= output.gradient

        output._backward = sub_grad
        return output

    def __mul__(self, other):
        other = other if isinstance(other, Node) else Node(other)
        output = Node(self.data * other.data, children=(self,other), operator='*')

        def mul_grad():
            self.gradient += other.data * output.gradient
            other.gradient += self.data * output.gradient

        output._backward = mul_grad
        return output

    def __truediv__(self, other):
        other = other if isinstance(other, Node) else Node(other)
        output = Node(self.data / other.data, children=(self,other), operator='/')

        def truediv_grad():
            self.gradient += (1/other.data) * output.gradient
            other.gradient += ((-self.data)/(other.data**2) * output.gradient)

        output._backward = truediv_grad
# The second _backward represents the function, it is a way to store a functon inside a variable simply without its brackets
        return output

    def __pow__(self, other):
        other = other if isinstance(other, Node) else Node(other)
        output = Node(self.data**other.data, children=(self,other), operator='^')

        def pow_grad():
            if self.data > 0:
                self.gradient += other.data*(self.data**(other.data - 1)) * output.gradient
                other.gradient += (self.data**other.data) * math.log(self.data) * output.gradient # e is default base
            elif self.data < 0 and other.data.is_integer():
                self.gradient -= other.data*(abs(self.data)**(other.data - 1)) * output.gradient
                other.gradient += (self.data**other.data) * math.log(abs(self.data)) * output.gradient
            elif self.data < 0 and not other.data.is_integer():
                raise ValueError(f"Your base, {self} is negative while your exponent, {other} is fractional, float, non-integer hence it is an imaginary number that is impossible to compute.")
            elif self.data == 0:
                if other.data >= 1:
                    self.gradient += 0
                    other.gradient += 0
                else:
                    raise ValueError(f"Your base, {self} is zero powered by your exponent, {other} is zero, you are trying to divide 0 by 0 that is impossible to compute.")

        output._backward = pow_grad
        return output

    # Reverse Operators
    def __radd__(self, other):
        return self + other # Calls __add__ function

    def __rsub__(self, other):
        other = other if isinstance(other, Node) else Node(other)
        output = Node(other.data - self.data, children=(other, self), operator='-')

        def sub_grad():
            other.gradient += output.gradient
            self.gradient -= output.gradient
        
        output._backward = sub_grad
        return output

    def __rmul__(self, other):
        return self * other # Calls __mul__ function

    def __rtruediv__(self, other):
        other = other if isinstance(other, Node) else Node(other)
        output = Node(other.data / self.data, children=(other, self), operator='/')

        def truediv_grad():
            other.gradient += (1/self.data) * output.gradient
            self.gradient += ((-other.data)/(self.data**2) * output.gradient)

        output._backward = truediv_grad
        return output

    def __rpow__(self, other):
        other = other if isinstance(other, Node) else Node(other)
        output = Node(other.data**self.data, children=(other, self), operator='^')

        def pow_grad():
            if other.data > 0:
                other.gradient += self.data*(other.data**(self.data-1)) * output.gradient
                self.gradient += (other.data**self.data) * math.log(other.data) * output.gradient # e is default base
            elif other.data < 0 and self.data.is_integer():
                other.gradient -= self.data*(abs(other.data)**(self.data - 1)) * output.gradient
                self.gradient += (other.data**self.data) * math.log(abs(other.data)) * output.gradient
            elif other.data < 0 and not self.data.is_integer():
                raise ValueError(f"Your base, {other} is negative while your exponent, {self} is fractional, float, non-integer hence it is an imaginary number that is impossible to compute.")
            elif other.data == 0:
                if self.data >= 1:
                    other.gradient += 0
                    self.gradient += 0
                else:
                    raise ValueError(f"Your base, {other} is zero powered by your exponent, {self} is zero, you are trying to divide 0 by 0 that is impossible to compute.")

        output._backward = pow_grad
        return output


    # Unary Operators

    def __abs__(self):
        output = Node(abs(self.data), children=(self,), operator='abs')

        def abs_grad():
            self.gradient += 0 if self.data == 0 else math.copysign(1, self.data) * output.gradient

        output._backward = abs_grad
        return output

    def e(self):
        output = Node(math.exp(self.data), children=(self,), operator='e')

        def e_grad():
            self.gradient += output.data * output.gradient

        output._backward = e_grad
        return output

    def __neg__(self):
        output = Node(-self.data, children=(self,), operator='neg')

        def neg_grad():
            self.gradient -= output.gradient

        output._backward = neg_grad
        return output

    
    # Activation function

    def tanh(self):
        output = Node(math.tanh(self.data), children=(self,), operator='tanh')

        def tanh_grad():
            self.gradient += (1 - output.data**2) * output.gradient

        output._backward = tanh_grad
        return output

    def relu(self, theta):
        if theta:
            return self.threshold_relu(theta)

        output = Node(max(0.0, self.data), children=(self,), operator='relu')

        def relu_grad():
            self.gradient += 1 * output.gradient if self.data > 0 else 0 

        output._backward = relu_grad
        return output

    def threshold_relu(self, theta=0):
        output = Node(self.data if self.data > 0 else 0, children=(self,), operator=f'threshold relu {theta}')

        def threshold_relu_grad():
            self.gradient += 1 * output.gradient if self.data > 0 else 0

        output._backward = threshold_relu_grad
        return output

    def prelu(self, alpha=0.25):
        if self.data >= 0: 
            val = self.data
            children = (self,)
            alpha_node = None
        else:
            alpha_node = Node(alpha, children=(), operator='alpha')
            val = alpha_node.data * self.data
            children = (self, alpha_node)
        
        output = Node(val, children=children, operator='prelu')  

        def prelu_grad():
            if self.data >= 0:
                self.gradient += 1 * output.gradient
            else:
                self.gradient += alpha_node.data * output.gradient
                alpha_node.gradient += self.data * output.gradient

        output._backward = prelu_grad
        return output

    def leaky_relu(self, a=0.01):
        output = Node(max(a*self.data, self.data), children=(self,), operator='leakyrelu')

        def leaky_relu_grad():
            self.gradient += 1 * output.gradient if self.data > 0 else a * output.gradient

        output._backward = leaky_relu_grad
        return output

    def sigmoid(self):
        output = Node(1/(1 + math.exp(-self.data)), children=(self,), operator='sigmoid')

        def sigmoid_grad():
            self.gradient += output.data*(1 - output.data) * output.gradient

        output._backward = sigmoid_grad
        return output

    def silu(self):
        output = Node(self.data/(1 + math.exp(-self.data)), children=(self,), operator='silu')

        def silu_grad():
            self.gradient += (1/(1 + math.exp(-self.data)) + self.data*(1/(1 + math.exp(-self.data)))*(1 - 1/(1 + math.exp(-self.data)))) * output.gradient

        output._backward = silu_grad
        return output

    def elu(self, alpha=1):
        output = Node(self.data if self.data > 0 else alpha*(math.exp(self.data) - 1), children=(self,), operator='elu')

        def elu_grad():
            self.gradient += 1 * output.gradient if self.data > 0 else (output.data + alpha) * output.gradient 

        output._backward = elu_grad
        return output

    def gelu(self):
        output = Node((0.5*self.data)*(1 + math.erf(self.data/2**0.5)), children=(self,), operator='gelu')

        def gelu_grad():
            cdf = 0.5 * (1 + math.erf(self.data/2**0.5)) # Cumalative Distribution Function
            pdf = (1/(2*math.pi)**0.5) * math.exp(-(self.data**2)/2) # Standard Normal Probability Function
            self.gradient += (cdf + self.data*pdf) * output.gradient
        output._backward = gelu_grad
        return output

    def selu(self, lam=1.0507 , a=1.67326):
        output = Node(lam*self.data if self.data > 0 else (lam*a*math.exp(self.data) - a), children=(self,), operator='selu')

        def selu_grad():
            self.gradient += lam * output.gradient if self.data > 0 else (output.data + lam*a) * output.gradient

        output._backward = selu_grad
        return output

    def softplus(self):
        output = Node(math.log(1 + math.exp(self.data)), children=(self,), operator='softplus')

        def softplus_grad():
            self.gradient += (1/(1 + math.exp(-self.data))) * output.gradient

        output._backward = softplus_grad
        return output

    def mish(self):
        output = Node(self.data*math.tanh(math.log(1 + math.exp(self.data))), children=(self,), operator='mish')

        def mish_grad():
            self.gradient += (math.tanh(math.log(1 + math.exp(self.data))) + self.data*(1 - (math.tanh(math.log(1 + math.exp(self.data))))**2)*(1/(1 + math.exp(-self.data)))) * output.gradient

        output._backward = mish_grad
        return output

    def hardswish(self):
        output = Node(self.data * (min(max(0, self.data + 3), 6)/6), children=(self,), operator='hardswish')

        def hardswish_grad():
            if self.data < -3:
                self.gradient += 0
            elif -3 <= self.data <= 3:
                self.gradient += ((2*self.data + 3)/3) * output.gradient
            elif self.data > 3:
                self.gradient += 1 * output.gradient

        output._backward = hardswish_grad
        return output

    


    def backward(self):
        self.gradient = 1
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v.children:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        for node in reversed(topo):
            if node._backward:
                node._backward()


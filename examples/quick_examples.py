from autograd import Value

# Level 1: Basic arithmetic operations
print("=== Basic Arithmetic ===")
a = Value(3.0)
b = Value(2.0)
c = a + b           # Addition
d = a - b           # Subtraction  
e = a * b           # Multiplication
f = a / b           # Division
print(f"3 + 2 = {c.data}, 3 - 2 = {d.data}, 3 * 2 = {e.data}, 3 / 2 = {f.data}")

# Level 2: Power operations and exponentials
print("\n=== Power & Exponentials ===")
x = Value(2.0)
y1 = x**3           # Scalar power: 2^3 = 8
y2 = x.exp()        # Natural exponential: e^2
print(f"2^3 = {y1.data:.2f}, e^2 = {y2.data:.2f}")

# Level 3: Logarithms and advanced powers
print("\n=== Logarithms & Value^Value ===")
z = Value(8.0)
w1 = z.log()        # Natural log: ln(8)
base = Value(2.0)
exp = Value(3.0)
w2 = base**exp      # Value^Value: 2^3 using exp(3*ln(2))
print(f"ln(8) = {w1.data:.2f}, 2^3 = {w2.data:.2f}")

# Level 4: Trigonometric functions
print("\n=== Trigonometric Functions ===")
angle = Value(1.0)  # 1 radian ≈ 57.3 degrees
t1 = angle.sin()
t2 = angle.cos() 
t3 = angle.tan()
print(f"sin(1) = {t1.data:.3f}, cos(1) = {t2.data:.3f}, tan(1) = {t3.data:.3f}")

# Level 5: Activation functions
print("\n=== Activation Functions ===")
neuron_input = Value(-0.5)
a1 = neuron_input.tanh()  # Hyperbolic tangent
a2 = neuron_input.relu()  # ReLU activation
print(f"tanh(-0.5) = {a1.data:.3f}, relu(-0.5) = {a2.data:.3f}")

# Level 6: Complex composition with gradients
print("\n=== Complex Composition ===")
x = Value(0.5)
# f(x) = sin(x^2) * exp(cos(x)) + ln(x + 1)
part1 = (x**2).sin()
part2 = x.cos().exp()
part3 = (x + 1).log()
result = part1 * part2 + part3

result.backward()
print(f"f(0.5) = {result.data:.4f}")
print(f"f'(0.5) = {x.grad:.4f}")

# Level 7: Multi-variable function with cross-dependencies
print("\n=== Multi-variable with Cross-dependencies ===")
# Reset gradients
for val in [a, b]:
    val.grad = 0.0

# g(a,b) = (a*b)^(a/b) * sin(a+b) + exp(a-b)
x1 = Value(2.0)
x2 = Value(4.0)

term1 = (x1 * x2)**(x1 / x2)  # (2*4)^(2/4) = 8^0.5
term2 = (x1 + x2).sin()       # sin(6)
term3 = (x1 - x2).exp()       # exp(-2)
final = term1 * term2 + term3

final.backward()
print(f"g(2,4) = {final.data:.4f}")
print(f"∂g/∂x1 = {x1.grad:.4f}, ∂g/∂x2 = {x2.grad:.4f}")

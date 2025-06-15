# Autograd

A lightweight automatic differentiation engine inspired by Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd). This project implements reverse-mode automatic differentiation (backpropagation) with a simple, educational focus on understanding the mechanics of neural network training.

## Overview

Autograd provides a scalar-valued automatic differentiation system that tracks computational graphs and computes gradients via backpropagation. It's designed to be minimal yet complete, making it perfect for learning how modern deep learning frameworks work under the hood.

## Key Features

- **Extended mathematical operations**: Beyond basic arithmetic, includes trigonometric functions (`sin`, `cos`, `tan`), natural logarithm (`log`), and advanced exponentiation (Value^Value)
- **Robust power operations**: Supports both scalar and Value exponents, with proper handling of edge cases
- **Comprehensive operator overloading**: Full support for Python's mathematical operators with proper broadcasting
- **Topological sorting**: Ensures correct gradient flow through complex computational graphs

## Core Operations

The `Value` class supports a comprehensive set of mathematical operations:
- **Arithmetic**: `+`, `-`, `*`, `/`, `**` (including Value^Value operations)
- **Activation functions**: `tanh()`, `relu()`, `exp()`
- **Trigonometric functions**: `sin()`, `cos()`, `tan()`
- **Logarithmic operations**: `log()` (natural logarithm)
- **Automatic gradient computation**: `.backward()` with topological sorting


## Examples

The `examples/` folder contains comprehensive demonstrations:

- **`quick_example.py`** - Basic operations and gradient computation
- **`neural_net.py`** - Complete neural network implementation using the Value class

See the examples directory for detailed code and explanations of how to build neural networks and train models using this autograd engine.


## Inspiration

This project is heavily inspired by [micrograd](https://github.com/karpathy/micrograd) by Andrej Karpathy. While micrograd focuses on being a minimal example, this implementation aims to expanded functionality while maintaining the same pedagogical clarity.


## Resources

- [micrograd](https://github.com/karpathy/micrograd) - The original inspiration
- [Automatic Differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation) - Wikipedia overview

---

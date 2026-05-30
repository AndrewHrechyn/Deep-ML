def poly_term_derivative(c: float, x: float, n: float) -> float:
    # Use the central difference formula to approximate the derivative
    h = 0.00001
    return ((c * pow((x + h), n)) - (c * pow((x - h), n))) / (2 * h)

def poly_term_derivative(c: float, x: float, n: float) -> float:
    # Use the left difference formula to approximate the derivative
    h = 0.00001
    return ((c * pow((x + h), n)) - (c * pow((x), n))) / h

def poly_term_derivative(c: float, x: float, n: float) -> float:
    # Use the right difference formula to approximate the derivative
    h = 0.00001
    return ((c * pow((x), n)) - (c * pow((x - h), n))) / h


import jax
import jax.numpy as jnp

def poly_term(c: float, x: float, n: float) -> float:
    return c * (x ** n)

def poly_term_derivative(c: float, x: float, n: float) -> float:

    poly_term_derivative_jax = jax.grad(poly_term, argnums=1)

    return poly_term_derivative_jax(c, x, n)


import torch 

def poly_term_derivative_torch(c: float, x: float, n: float) -> float:

    x = torch.tensor(float(x), requires_grad=True)

    y = c * (x ** n)

    y.backward()

    return x.grad.item()

from tinygrad.tensor import Tensor

def poly_term_derivative_tinygrad(c: float, x: float, n: float) -> float:

    x = Tensor(float(x), requires_grad=True)

    y = c * (x ** n)

    y.backward()

    return x.grad.numpy()[0]
import torch
import torch.nn as nn
import torch.nn.functional as F

import nn_arz.nn_arz.utils

class Polynomial(nn.Module):
    def __init__(self, coefficients, device='cpu'):
        super(Polynomial, self).__init__()

        self.coefficients = nn.Parameter(coefficients.to(device))
        self.device = device

    def to(self, device):
        self.device = device
        self.coefficients = nn.Parameter(self.coefficients.to(device))
        return self
    
    def update(self, coefficients):
        self.coefficients = nn.Parameter(coefficients.to(self.device))
        return self
    
    def __mul__(self, other):
        # Cauchy product 
        coefficients_A = self.coefficients
        coefficients_B = other.coefficients.to(self.device)

        n = coefficients_A.shape[0]
        m = coefficients_B.shape[0]

        # Extend the coefficients to the same size
        coefficients_A = torch.cat([coefficients_A, torch.zeros(m, device=self.device)], dim=0)
        coefficients_B = torch.cat([coefficients_B, torch.zeros(n, device=self.device)], dim=0)
                                             
        result = torch.zeros(n + m - 1, device=self.device)
        for i in range(n + m - 1):
            result[i] = torch.sum(coefficients_A[:i+1] * coefficients_B[:i+1].flip(0))

        return Polynomial(result, device=self.device)
    
    def __call__(self, x):
        x = x.unsqueeze(1)
        return x ** torch.arange(self.coefficients.shape[0], device=self.device) @ self.coefficients

    def __repr__(self):
        str = f'Polynomial('
        for i, coefficient in enumerate(self.coefficients):
            if coefficient > 0:
                str += f' + {coefficient.item()}X^{i}'
            elif coefficient < 0:
                str += f' - {-coefficient.item()}X^{i}'
        str += f', device={self.device})'
        return str
    
class BatchPolynomial(nn.Module):
    def __init__(self, coefficients, device='cpu', requires_grad=False):
        super(BatchPolynomial, self).__init__()
        self.requires_grad = requires_grad
        if requires_grad:
            self.coefficients = nn.Parameter(coefficients.to(device))
        else:
            self.coefficients = coefficients.to(device)
        self.device = device

    def to(self, device):
        self.device = device

        if self.requires_grad:
            self.coefficients = nn.Parameter(self.coefficients.to(device))
        else:
            self.coefficients = self.coefficients.to(device)
        return self

    
    def sum(self, weights=None):
        if weights is None:
            return Polynomial(torch.sum(self.coefficients, dim=0), device=self.device)
        return Polynomial(torch.sum(self.coefficients * weights.unsqueeze(1), dim=0), device=self.device)

    def prime(self):
        return BatchPolynomial(self.coefficients[:, 1:] * torch.arange(1, self.coefficients.shape[1], device=self.device), device=self.device)

    def antiderivative(self):
        return BatchPolynomial(torch.cat([torch.zeros(self.coefficients.shape[0], 1, device=self.device), self.coefficients / torch.arange(1, self.coefficients.shape[1] + 1, device=self.device)], dim=1), device=self.device)
        # return BatchPolynomial(torch.cat([torch.zeros(self.coefficients.shape[0], 1, device=self.device), self.coefficients / torch.arange(1, self.coefficients.shape[1], device=self.device)], dim=1), device=self.device)
    
    def __mul__(self, other):
        if isinstance(other, BatchPolynomial):
            # Cauchy product 
            coefficients_A = self.coefficients
            coefficients_B = other.coefficients.to(self.device)
            if coefficients_A.shape[0] != coefficients_B.shape[0]:
                raise ValueError("The batch sizes must be the same.")

            n = coefficients_A.shape[1]
            m = coefficients_B.shape[1]
            batch_size = coefficients_A.shape[0]

            # Extend the coefficients to the same size
            coefficients_A = torch.cat([coefficients_A, torch.zeros((batch_size, m), device=self.device)], dim=1)
            coefficients_B = torch.cat([coefficients_B, torch.zeros((batch_size, n), device=self.device)], dim=1)
                                                
            result = torch.zeros((batch_size, n + m - 1), device=self.device)
            for i in range(n + m - 1):
                result[:, i] = torch.sum(coefficients_A[:, :i+1] * coefficients_B[:, :i+1].flip(1), dim=1)

            return BatchPolynomial(result, device=self.device)

    def __call__(self, x, max_degree=None):
        x = torch.stack([x] * self.coefficients.shape[0]).unsqueeze(-1).transpose(0, 1)

        power = torch.arange(self.coefficients.shape[1])
        power = torch.stack([power] * x.shape[1]).unsqueeze(0).to(device=self.device)

        # mask = (power <= max_degree if max_degree is not None else torch.ones_like(power))[0]
        return (((x**power) * self.coefficients.unsqueeze(0)).transpose(0, 1)).sum(dim=-1)
    
    def __repr__(self):
        str = ''
        for Polynomial in self.coefficients:
            str += f'Polynomial('
            for i, coefficient in enumerate(Polynomial):
                if coefficient > 0:
                    str += f' + {coefficient.item()}X^{i}'
                elif coefficient < 0:
                    str += f' - {-coefficient.item()}X^{i}'
            str += f', device={self.device})'
            str += '\n'
        return str[:-1]
    
def legendreCoefficients(max_degree):
    # Initialize the coefficient matrix (max_degree + 1) x (max_degree + 1)
    coeffs = torch.zeros((max_degree + 1, max_degree + 1), dtype=torch.float32)

    # Initial conditions: P_0(x) = 1 and P_1(x) = x
    coeffs[0, 0] = 1  # P_0(x) = 1
    if max_degree == 0:
        return coeffs
    coeffs[1, 1] = 1  # P_1(x) = x

    # Recurrence relation to fill the matrix:
    # (n + 1)P_{n+1}(x) = (2n + 1)xP_n(x) - nP_{n-1}(x)
    for n in range(1, max_degree):
        coeffs[n + 1, 1:] += (2 * n + 1) * coeffs[n, :-1]  # Multiply P_n(x) by x
        coeffs[n + 1, :] -= n * coeffs[n - 1, :]  # Subtract n * P_{n-1}(x)
        coeffs[n + 1, :] /= (n + 1)  # Divide by (n + 1)

    return coeffs

def bernsteinCoefficients(degree):
    coeffs = torch.zeros((degree + 1, degree + 1), dtype=torch.float32)
    
    coeffs[-1, 0] = 1
    for nu in range(degree - 1, -1, -1):
        coeffs[nu] = coeffs[nu + 1] - torch.roll(coeffs[nu + 1], 1)

    binoms = nn_arz.nn_arz.utils.torch_binomial_coefficient(degree * torch.ones(degree+1), torch.arange(degree + 1))
    for nu in range(degree + 1):
        coeffs[nu] = torch.roll(coeffs[nu], nu) * binoms[nu]

    return coeffs

def tchebychevCoefficients(degree):
    coeffs = torch.zeros((degree + 1, degree + 1), dtype=torch.float32)
    coeffs[0, 0] = 1
    if degree == 0:
        return coeffs
    coeffs[1, 1] = 1

    for n in range(1, degree):
        coeffs[n + 1, 1:] += 2 * coeffs[n, :-1]
        coeffs[n + 1, :] -= coeffs[n - 1, :]

    return coeffs

# def legendreBasis(degree, requires_grad=False):
#     coeffs = nn_arz.nn_arz.polynomials.legendreCoefficients(15)
#     legendre = nn_arz.nn_arz.polynomials.BatchPolynomial(coeffs)
#     legendre.coefficients.requires_grad = False
#     return legendre
    
def bernsteinBasis(degree, requires_grad=False):
    coeffs = nn_arz.nn_arz.polynomials.bernsteinCoefficients(15)
    bernstein = nn_arz.nn_arz.polynomials.BatchPolynomial(coeffs)
    bernstein.coefficients.requires_grad = False
    return bernstein

def tchebychevBasis(degree, requires_grad=False):
    coeffs = nn_arz.nn_arz.polynomials.tchebychevCoefficients(15)
    tchebychev = nn_arz.nn_arz.polynomials.BatchPolynomial(coeffs)
    tchebychev.coefficients.requires_grad = False
    return tchebychev

class LegendreBasis(BatchPolynomial):
    def __init__(self, degree, requires_grad=False, device='cpu'):
        coeffs = legendreCoefficients(degree)
        self.degree = degree
        super().__init__(coefficients=coeffs, requires_grad=requires_grad, device=device)

    def mass_matrix(self):
        return torch.tensor([2 / (2 * i + 1) for i in range(self.degree+1)]).to(self.device)

def basis(polynomials, number_of_polynomials, device='cpu'):
    if polynomials == 'chebyshev':
        raise NotImplementedError
    elif polynomials == 'bernstein':
        raise NotImplementedError 
    elif polynomials == 'legendre':
        return LegendreBasis(number_of_polynomials - 1, device=device)
    else:
        raise NotImplementedError
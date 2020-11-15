import numpy as np
import torch


FLOAT_EPSILON = 1e-8


def flat_concat(xs):
    return torch.cat([torch.reshape(x, (-1,)) for x in xs], dim=0)


def assign_params_from_flat(new_params, params):
    def flat_size(p):
        return int(np.prod(p.shape))
    splits = torch.split(new_params, [flat_size(p) for p in params])
    new_params = [torch.reshape(p_new, p.shape)
                  for p, p_new in zip(params, splits)]
    for p, p_new in zip(params, new_params):
        p.data.copy_(p_new)


class ConjugateGradient:
    def __init__(
        self, conjugate_gradient_steps=10, damping_coefficient=0.1,
        constraint_threshold=0.01, backtrack_steps=10,
        backtrack_coefficient=0.8
    ):
        self.conjugate_gradient_steps = conjugate_gradient_steps
        self.damping_coefficient = damping_coefficient
        self.constraint_threshold = constraint_threshold
        self.backtrack_steps = backtrack_steps
        self.backtrack_coefficient = backtrack_coefficient

    def optimize(self, loss_function, constraint_function, variables):
        def _hx(x):
            f = constraint_function()
            gradient_1 = torch.autograd.grad(f, variables, create_graph=True)
            gradient_1 = flat_concat(gradient_1)
            x = torch.as_tensor(x)
            y = (gradient_1 * x).sum()
            gradient_2 = torch.autograd.grad(y, variables)
            gradient_2 = flat_concat(gradient_2)

            if self.damping_coefficient > 0:
                gradient_2 += self.damping_coefficient * x

            return gradient_2

        def _cg(b):
            x = np.zeros_like(b)
            r = b.copy()
            p = r.copy()
            r_dot_old = np.dot(r, r)
            if r_dot_old == 0:
                return None

            for _ in range(self.conjugate_gradient_steps):
                z = _hx(p).numpy()
                alpha = r_dot_old / (np.dot(p, z) + FLOAT_EPSILON)
                x += alpha * p
                r -= alpha * z
                r_dot_new = np.dot(r, r)
                p = r + (r_dot_new / r_dot_old) * p
                r_dot_old = r_dot_new
            return x

        def _update(alpha, conjugate_gradient, step, start_variables):
            conjugate_gradient = torch.as_tensor(conjugate_gradient)
            new_variables = start_variables - alpha * conjugate_gradient * step
            assign_params_from_flat(new_variables, variables)
            constraint = constraint_function()
            loss = loss_function()
            return constraint.detach(), loss.detach()

        start_variables = flat_concat(variables)

        for var in variables:
            if var.grad:
                var.grad.data.zero_()

        loss = loss_function()
        grad = torch.autograd.grad(loss, variables)
        grad = flat_concat(grad).numpy()
        start_loss = loss.detach().numpy()

        conjugate_gradient = _cg(grad)
        if conjugate_gradient is None:
            constraint = torch.as_tensor(0., dtype=torch.float32)
            loss = torch.as_tensor(0., dtype=torch.float32)
            steps = torch.as_tensor(0, dtype=torch.int32)
            return constraint, loss, steps

        alpha = np.sqrt(2 * self.constraint_threshold / np.dot(
            conjugate_gradient, _hx(conjugate_gradient)) + FLOAT_EPSILON)

        if self.backtrack_steps is None or self.backtrack_coefficient is None:
            constraint, loss = _update(
                alpha, conjugate_gradient, 1, start_variables)
            return constraint, loss

        for i in range(self.backtrack_steps):
            constraint, loss = _update(
                alpha, conjugate_gradient, self.backtrack_coefficient ** i,
                start_variables)

            if (constraint.numpy() <= self.constraint_threshold and
                    loss.numpy() <= start_loss):
                break

            if i == self.backtrack_steps - 1:
                constraint, loss = _update(
                    alpha, conjugate_gradient, 0, start_variables)
                i = self.backtrack_steps

        return constraint, loss, torch.as_tensor(i + 1, dtype=torch.int32)

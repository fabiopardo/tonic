import numpy as np
import tensorflow as tf


FLOAT_EPSILON = 1e-8


def flat_concat(xs):
    return tf.concat([tf.reshape(x, (-1,)) for x in xs], axis=0)


def assign_params_from_flat(new_params, params):
    def flat_size(p):
        return int(np.prod(p.shape.as_list()))
    splits = tf.split(new_params, [flat_size(p) for p in params])
    new_params = [tf.reshape(p_new, p.shape)
                  for p, p_new in zip(params, splits)]
    for p, p_new in zip(params, new_params):
        p.assign(p_new)


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
        @tf.function
        def _hx(x):
            with tf.GradientTape() as tape_2:
                with tf.GradientTape() as tape_1:
                    f = constraint_function()
                gradient_1 = flat_concat(tape_1.gradient(f, variables))
                y = tf.reduce_sum(gradient_1 * x)
            gradient_2 = flat_concat(tape_2.gradient(y, variables))

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

        @tf.function
        def _update(alpha, conjugate_gradient, step, start_variables):
            new_variables = start_variables - alpha * conjugate_gradient * step
            assign_params_from_flat(new_variables, variables)
            constraint = constraint_function()
            loss = loss_function()
            return constraint, loss

        start_variables = flat_concat(variables)

        with tf.GradientTape() as tape:
            loss = loss_function()
        grad = flat_concat(tape.gradient(loss, variables)).numpy()
        start_loss = loss.numpy()

        conjugate_gradient = _cg(grad)
        if conjugate_gradient is None:
            constraint = tf.convert_to_tensor(0.)
            loss = tf.convert_to_tensor(0.)
            steps = tf.convert_to_tensor(0)
            return constraint, loss, steps

        alpha = np.sqrt(2 * self.constraint_threshold / np.dot(
            conjugate_gradient, _hx(conjugate_gradient)) + FLOAT_EPSILON)
        alpha = tf.convert_to_tensor(alpha, tf.float32)

        if self.backtrack_steps is None or self.backtrack_coefficient is None:
            constraint, loss = _update(
                alpha, conjugate_gradient, 1, start_variables)
            return constraint, loss

        for i in range(self.backtrack_steps):
            step = tf.convert_to_tensor(
                self.backtrack_coefficient ** i, tf.float32)
            constraint, loss = _update(
                alpha, conjugate_gradient, step, start_variables)

            if constraint <= self.constraint_threshold and loss <= start_loss:
                break

            if i == self.backtrack_steps - 1:
                step = tf.convert_to_tensor(0., tf.float32)
                constraint, loss = _update(
                    alpha, conjugate_gradient, step, start_variables)
                i = self.backtrack_steps

        return constraint, loss, tf.convert_to_tensor(i + 1, dtype=tf.int32)

def tile(x, n):
    return x[None].repeat([n] + [1] * len(x.shape))


def merge_first_two_dims(x):
    return x.view(x.shape[0] * x.shape[1], *x.shape[2:])

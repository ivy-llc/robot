"""
Spline path sampling functions, useful for differentiable continuous robot path planning.
"""

# global
from ivy.framework_handler import get_framework as _get_framework


# Helpers #
# --------#

def _pairwise_distance(x, y, f):

    # BS x NX x 1 x 1
    x = f.expand_dims(x, -2)

    # BS x 1 x NY x 1
    y = f.expand_dims(y, -3)

    # BS x NX x NY
    return f.reduce_sum((x - y) ** 2, -1)


def _phi(r, order, f):
    eps = f.array([1e-6], 'float32')
    if order % 2 == 0:
        r = f.maximum(r, eps)
        return 0.5 * (r ** (0.5 * order)) * f.log(r)
    else:
        r = f.maximum(r, eps)
        return r ** (0.5 * order)


def _fit_spline(train_points, train_values, order, f):

    # shapes
    train_points_shape = train_points.shape
    batch_shape = list(train_points_shape[:-2])
    num_batch_dims = len(batch_shape)
    n = train_points_shape[-2]
    pd = train_values.shape[-1]

    # BS x N x 1
    c = train_points

    # BS x N x PD
    f_ = train_values

    # BS x N x N
    matrix_a = _phi(_pairwise_distance(c, c, f), order, f)

    # BS x N x 1
    ones = f.ones_like(c[..., :1])

    # BS x N x 2
    matrix_b = f.concatenate([c, ones], -1)

    # BS x 2 x N
    matrix_b_trans = f.transpose(matrix_b, list(range(num_batch_dims)) + [num_batch_dims + 1, num_batch_dims])

    # BS x N+2 x N
    left_block = f.concatenate([matrix_a, matrix_b_trans], -2)

    # BS x 2 x 2
    lhs_zeros = f.zeros(batch_shape + [2, 2])

    # BS x N+2 x 2
    right_block = f.concatenate([matrix_b, lhs_zeros], -2)

    # BS x N+2 x N+2
    lhs = f.concatenate([left_block, right_block], -1)

    # BS x 2 x PD
    rhs_zeros = f.zeros(batch_shape + [2, pd])

    # BS x N+2 x PD
    rhs = f.concatenate([f_, rhs_zeros], -2)

    # BS x N+2 x PD
    w_v = f.matmul(f.pinv(lhs), rhs)

    # BS x N x PD
    w = w_v[..., :n, :]

    # BS x 2 x PD
    v = w_v[..., n:, :]

    # BS x N x PD,    BS x 2 x PD
    return w, v


# Public #
# -------#

def sample_spline_path(anchor_points, anchor_vals, sample_points, order=3, f=None):
    """
    Sample spline path, given sample locations, for path defined by the anchor locations and points.
    `[reference] <https://github.com/tensorflow/addons/blob/v0.11.2/tensorflow_addons/image/interpolate_spline.py>`_

    :param anchor_points: Anchor locations between 0-1 (regular spacing not necessary) *[batch_shape,num_anchors,1]*
    :type anchor_points: array
    :param anchor_vals: Anchor points along the spline path, in path space *[batch_shape,num_anchors,path_dim]*
    :type anchor_vals: array
    :param sample_points: Sample locations between 0-1 *[batch_shape,num_samples,1]*
    :type sample_points: array
    :param order: Order of the spline path interpolation
    :type order: float
    :param f: Machine learning library. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Spline path sampled at sample_locations, giving points in path space *[batch_shape,num_samples,path_dim]*
    """
    f = _get_framework(sample_points, f=f)

    # BS x N x PD,    BS x 2 x PD
    w, v = _fit_spline(anchor_points, anchor_vals, order, f)

    # Kernel term

    # BS x NS x N
    pairwise_dists = _pairwise_distance(sample_points, anchor_points, f)
    phi_pairwise_dists = _phi(pairwise_dists, order, f)

    # BS x NS x PD
    rbf_term = f.matmul(phi_pairwise_dists, w)

    # Polynomial / linear term.

    # BS x NS x 2
    query_points_pad = f.concatenate([sample_points, f.ones_like(sample_points[..., :1])], -1)

    # BS x NS x PD
    linear_term = f.matmul(query_points_pad, v)
    return rbf_term + linear_term

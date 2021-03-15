"""
Spline path sampling functions, useful for differentiable continuous robot path planning.
"""

# global
import ivy as _ivy


# Helpers #
# --------#

def _pairwise_distance(x, y):

    # BS x NX x 1 x 1
    try:
        x = _ivy.expand_dims(x, -2)
    except:
        d = 0

    # BS x 1 x NY x 1
    y = _ivy.expand_dims(y, -3)

    # BS x NX x NY
    return _ivy.reduce_sum((x - y) ** 2, -1)


def _phi(r, order):
    eps = _ivy.array([1e-6], 'float32')
    if order % 2 == 0:
        r = _ivy.maximum(r, eps)
        return 0.5 * (r ** (0.5 * order)) * _ivy.log(r)
    else:
        r = _ivy.maximum(r, eps)
        return r ** (0.5 * order)


def _fit_spline(train_points, train_values, order):

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
    matrix_a = _phi(_pairwise_distance(c, c), order)

    # BS x N x 1
    ones = _ivy.ones_like(c[..., :1])

    # BS x N x 2
    matrix_b = _ivy.concatenate([c, ones], -1)

    # BS x 2 x N
    matrix_b_trans = _ivy.transpose(matrix_b, list(range(num_batch_dims)) + [num_batch_dims + 1, num_batch_dims])

    # BS x N+2 x N
    left_block = _ivy.concatenate([matrix_a, matrix_b_trans], -2)

    # BS x 2 x 2
    lhs_zeros = _ivy.zeros(batch_shape + [2, 2])

    # BS x N+2 x 2
    right_block = _ivy.concatenate([matrix_b, lhs_zeros], -2)

    # BS x N+2 x N+2
    lhs = _ivy.concatenate([left_block, right_block], -1)

    # BS x 2 x PD
    rhs_zeros = _ivy.zeros(batch_shape + [2, pd])

    # BS x N+2 x PD
    rhs = _ivy.concatenate([f_, rhs_zeros], -2)

    # BS x N+2 x PD
    w_v = _ivy.matmul(_ivy.pinv(lhs), rhs)

    # BS x N x PD
    w = w_v[..., :n, :]

    # BS x 2 x PD
    v = w_v[..., n:, :]

    # BS x N x PD,    BS x 2 x PD
    return w, v


# Public #
# -------#

def sample_spline_path(anchor_points, anchor_vals, sample_points, order=3):
    """
    Sample spline path, given sample locations for path defined by the anchor locations and points.
    `[reference] <https://github.com/tensorflow/addons/blob/v0.11.2/tensorflow_addons/image/interpolate_spline.py>`_

    :param anchor_points: Anchor locations between 0-1 (regular spacing not necessary) *[batch_shape,num_anchors,1]*
    :type anchor_points: array
    :param anchor_vals: Anchor points along the spline path, in path space *[batch_shape,num_anchors,path_dim]*
    :type anchor_vals: array
    :param sample_points: Sample locations between 0-1 *[batch_shape,num_samples,1]*
    :type sample_points: array
    :param order: Order of the spline path interpolation
    :type order: float
    :return: Spline path sampled at sample_locations, giving points in path space *[batch_shape,num_samples,path_dim]*
    """

    # BS x N x PD,    BS x 2 x PD
    w, v = _fit_spline(anchor_points, anchor_vals, order)

    # Kernel term

    # BS x NS x N
    pairwise_dists = _pairwise_distance(sample_points, anchor_points)
    phi_pairwise_dists = _phi(pairwise_dists, order)

    # BS x NS x PD
    rbf_term = _ivy.matmul(phi_pairwise_dists, w)

    # Polynomial / linear term.

    # BS x NS x 2
    query_points_pad = _ivy.concatenate([sample_points, _ivy.ones_like(sample_points[..., :1])], -1)

    # BS x NS x PD
    linear_term = _ivy.matmul(query_points_pad, v)
    return rbf_term + linear_term

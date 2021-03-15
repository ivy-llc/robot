"""
Rigid Mobile class, containing functions for sampling rigid mobile robots
"""

# global
import ivy_mech
import ivy as _ivy

MIN_DENOMINATOR = 1e-12


class RigidMobile:

    def __init__(self, rel_body_points):
        """
        Initialize rigid mobile robot instance

        :param rel_body_points: Relative body points *[num_body_points,3]*
        :type rel_body_points: array
        """

        # 4 x NBP
        self._rel_body_points_homo_trans =\
            _ivy.swapaxes(ivy_mech.make_coordinates_homogeneous(rel_body_points), 0, 1)

    # Public Methods #
    # ---------------#

    # Body sampling #

    def sample_body(self, inv_ext_mats, batch_shape=None):
        """
        Sample links of the robot at uniformly distributed cartesian positions.

        :param inv_ext_mats: Inverse extrinsic matrices *[batch_shape,3,4]*
        :type inv_ext_mats: array
        :param batch_shape: Shape of batch. Inferred from inputs if None.
        :type batch_shape: sequence of ints, optional
        :return: The sampled body cartesian positions, in the world reference frame *[batch_shape,num_body_points,3]*
        """

        if batch_shape is None:
            batch_shape = inv_ext_mats.shape[:-2]
        batch_shape = list(batch_shape)

        # (BSx3) x NBP
        body_points_trans = _ivy.matmul(_ivy.reshape(inv_ext_mats, (-1, 4)), self._rel_body_points_homo_trans)

        # BS x NBP x 3
        return _ivy.swapaxes(_ivy.reshape(body_points_trans, batch_shape + [3, -1]), -1, -2)

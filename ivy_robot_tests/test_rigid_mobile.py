"""
Collection of tests for mico robot manipulator
"""

# global
import ivy_mech
import numpy as np

# local
import ivy
import ivy_robot_tests.helpers as helpers
from ivy_robot.rigid_mobile import RigidMobile


class RigidMobileTestData:

    def __init__(self):
        rot_vec_pose = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        self.inv_ext_mat = ivy_mech.rot_vec_pose_to_mat_pose(rot_vec_pose)
        self.rel_body_points = np.array([[0., 0., 0.],
                                         [-0.2, -0.2, 0.],
                                         [-0.2, 0.2, 0.],
                                         [0.2, -0.2, 0.],
                                         [0.2, 0.2, 0.]])
        self.sampled_body = np.array([[0.1, 0.2, 0.3],
                                      [0.04361792, -0.0751835, 0.26690764],
                                      [-0.12924806, 0.22732089, 0.46339797],
                                      [0.32924806, 0.17267911, 0.13660203],
                                      [0.15638208, 0.4751835, 0.33309236]])


td = RigidMobileTestData()


def test_sample_body():
    for lib, call in helpers.calls:
        if call is helpers.mx_graph_call:
            # mxnet symbolic does not fully support array slicing
            continue
        mico = RigidMobile(lib.array(td.rel_body_points, 'float32'), lib)
        assert np.allclose(call(mico.sample_body, td.inv_ext_mat),
                           td.sampled_body, atol=1e-6)
        assert np.allclose(call(mico.sample_body, ivy.tile(ivy.expand_dims(td.inv_ext_mat, 0), (5, 1, 1))),
                           ivy.tile(ivy.expand_dims(td.sampled_body, 0), (5, 1, 1)), atol=1e-6)

"""
Collection of tests for unified linear algebra functions
"""

# global
import ivy
import numpy as np

# local
import ivy_robot


class SplineTestData:

    def __init__(self):
        # 3d example
        self.train_points_3d = np.array([[0.], [0.25], [0.5], [0.75], [1.]]).astype(np.float32)
        self.train_values_3d = np.array([[-1.15, -1.028, 0.6],
                                         [-0.65612626, -0.46623456, 0.65002733],
                                         [-0.11246832, 0.09845974, 0.650022],
                                         [0.43124327, 0.63675475, 0.65001863],
                                         [1.025, 1.125, 0.6]]).astype(np.float32)
        self.query_points_3d = np.array([[1.]]).astype(np.float32)
        self.query_values_3d = np.array([[1.025, 1.125, 0.6]]).astype(np.float32)


td = SplineTestData()


def test_spline_interpolation(dev_str, fw):
    ivy.set_backend(fw)
    assert np.allclose(
        ivy_robot.planning.sample_spline_path(ivy.array(td.train_points_3d), ivy.array(td.train_values_3d),
             ivy.array(td.query_points_3d)), td.query_values_3d, atol=1e-2)
    ivy.unset_backend()

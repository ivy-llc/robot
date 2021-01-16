"""
Collection of tests for mico robot manipulator
"""

# global
import numpy as np

# local
import ivy
import ivy_robot_tests.helpers as helpers
from ivy_robot.manipulator import MicoManipulator


class MicoTestData:

    def __init__(self):
        # link matrix extraction

        # Generated using random joint angles in CoppeliaSim scene with robot model queried directly for matrices
        self.joint_angles = np.array([-0.3202435076236725,
                                      5.3120341300964355,
                                      4.000271320343018,
                                      1.116408348083496,
                                      -2.156550645828247,
                                      -1.7456011772155762])
        l0_mat = np.expand_dims(np.eye(4, 4), 0)
        l1_mat = np.array([[[0.949158787727356, 1.341104507446289e-07, 0.3147977888584137, 2.996603143401444e-08],
                            [0.3147977590560913, -3.5762786865234375e-07, -0.9491588473320007, -1.0186340659856796e-09],
                            [-2.9802322387695312e-08, 1.0000001192092896, -4.76837158203125e-07, 0.27549999952316284],
                            [0.0, 0.0, 0.0, 1.0]]])
        l2_mat = np.array([[[-0.7835646867752075, -0.5356574058532715, -0.3147978186607361, -0.22723382711410522],
                            [-0.25987696647644043, -0.17765593528747559, 0.949158787727356, -0.07536435127258301],
                            [-0.5643495917320251, 0.8255358934402466, -1.1920928955078125e-07, 0.11183865368366241],
                            [0.0, 0.0, 0.0, 1.0]]])
        l3_mat = np.array([[[-0.24312376976013184, -0.3147977292537689, 0.9174930453300476, -0.22502225637435913],
                            [-0.08063429594039917, 0.949158787727356, 0.30429545044898987, -0.0820058137178421],
                            [-0.9666381478309631, 0.0, -0.25614655017852783, 0.11187019944190979],
                            [0.0, 0.0, 0.0, 1.0]]])
        l4_mat = np.array([[[-0.3895663022994995, 0.8347147107124329, 0.38921672105789185, -0.37740737199783325],
                            [0.8174557089805603, 0.508053183555603, -0.2713821232318878, -0.13254578411579132],
                            [-0.424269437789917, 0.21244612336158752, -0.8802627325057983, 0.15437977015972137],
                            [0.0, 0.0, 0.0, 1.0]]])
        l5_mat = np.array([[[0.48020344972610474, 0.7301088571548462, -0.48615410923957825, -0.41064488887786865],
                            [0.8752689361572266, -0.43518805503845215, 0.21098817884922028, -0.10929147154092789],
                            [-0.05752405524253845, -0.5268328189849854, -0.8480204343795776, 0.22971707582473755],
                            [0.0, 0.0, 0.0, 1.0]]])
        l6_mat = np.array([[[0.3458806276321411, -0.8025201559066772, 0.48613572120666504, -0.31196799874305725],
                            [0.9376280307769775, 0.27634233236312866, -0.21092276275157928, -0.15230382978916168],
                            [0.034929946064949036, 0.5287685990333557, 0.8480469584465027, 0.40156668424606323],
                            [0.0, 0.0, 0.0, 1.0]]])

        self.true_link_matrices = np.concatenate((l0_mat, l1_mat, l2_mat, l3_mat, l4_mat, l5_mat, l6_mat), 0)

        # link sampling

        self.sampled_link = np.array([[0., 0., 0.],
                                      [0., 0., 0.04591667],
                                      [0., 0., 0.09183333],
                                      [0., 0., 0.13775],
                                      [0., 0., 0.18366667],
                                      [0., 0., 0.22958333],
                                      [0., 0., 0.2755],
                                      [-0.03246197, -0.01076632, 0.2521198],
                                      [-0.06492393, -0.02153265, 0.22873961],
                                      [-0.0973859, -0.03229897, 0.20535941],
                                      [-0.12984786, -0.0430653, 0.18197921],
                                      [-0.16230983, -0.05383162, 0.15859902],
                                      [-0.19477179, -0.06459795, 0.13521882],
                                      [-0.22723376, -0.07536427, 0.11183863],
                                      [-0.22503017, -0.08200839, 0.11183863],
                                      [-0.26312486, -0.09464286, 0.12247393],
                                      [-0.30121955, -0.10727733, 0.13310924],
                                      [-0.33931423, -0.1199118, 0.14374454],
                                      [-0.37740892, -0.13254627, 0.15437985],
                                      [-0.39406029, -0.1209361, 0.19203892],
                                      [-0.41071165, -0.10932593, 0.229698],
                                      [-0.39099505, -0.11788305, 0.26409055],
                                      [-0.37127844, -0.12644016, 0.29848309],
                                      [-0.35156184, -0.13499728,  0.33287564],
                                      [-0.33184523, -0.14355439, 0.36726818],
                                      [-0.31212863, -0.15211151, 0.40166073]])


td = MicoTestData()


def test_compute_mico_link_matrices():
    for lib, call in helpers.calls:
        if call is helpers.mx_graph_call:
            # mxnet symbolic does not fully support array slicing
            continue
        mico = MicoManipulator(lib)
        assert np.allclose(call(mico.compute_link_matrices, td.joint_angles, 6),
                           td.true_link_matrices, rtol=1e-03, atol=1e-03)
        assert np.allclose(call(mico.compute_link_matrices, ivy.tile(ivy.expand_dims(td.joint_angles, 0), (5, 1)), 6),
                           ivy.tile(ivy.expand_dims(td.true_link_matrices, 0), (5, 1, 1, 1)), rtol=1e-03, atol=1e-03)


def test_sample_mico_links():
    for lib, call in helpers.calls:
        if call in [helpers.tf_graph_call, helpers.mx_graph_call]:
            # the need to dynamically infer array shapes makes this only valid in eager mode currently
            continue
        mico = MicoManipulator(lib)
        assert np.allclose(call(mico.sample_links, td.joint_angles, 6),
                           td.sampled_link, atol=1e-6)
        assert np.allclose(call(mico.sample_links, ivy.tile(ivy.expand_dims(td.joint_angles, 0), (5, 1)), 6),
                           ivy.tile(ivy.expand_dims(td.sampled_link, 0), (5, 1, 1, 1)), atol=1e-6)

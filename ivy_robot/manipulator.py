"""
Robot Manipulator class, containing core kinematics and jacobian functions
"""

# global
import ivy_mech as _ivy_mech
from ivy.framework_handler import get_framework as _get_framework
import math as _math

MIN_DENOMINATOR = 1e-12


class Manipulator:

    def __init__(self, a_s, d_s, alpha_s, dh_joint_scales, dh_joint_offsets, base_inv_ext_mat=None, f=None):
        """
        Initialize robot manipulator instance

        :param a_s: Denavit–Hartenberg "a" parameters *[num_joints]*
        :type a_s: array
        :param d_s: Denavit–Hartenberg "d" parameters *[num_joints]*
        :type d_s: array
        :param alpha_s: Denavit–Hartenberg "alpha" parameters *[num_joints]*
        :type alpha_s: array
        :param dh_joint_scales: Scalars to apply to input joints *[num_joints]*
        :type dh_joint_scales: array
        :param dh_joint_offsets: Scalar offsets to apply to input joints *[num_joints]*
        :type dh_joint_offsets: array
        :param base_inv_ext_mat: Inverse extrinsic matrix of the robot base *[4,4]*
        :type base_inv_ext_mat: array, optional
        :param f: Machine learning framework. Inferred from inputs if None.
        :type f: ml_framework, optional
        """

        self._f = _get_framework(a_s, f=f)
        self._num_joints = a_s.shape[-1]
        # ToDo: incorporate the base_inv_ext_mat more elegantly, instead of the hack as in the sample_links method
        if base_inv_ext_mat is None:
            self._base_inv_ext_mat = self._f.identity(4)
        else:
            self._base_inv_ext_mat = base_inv_ext_mat

        # NJ
        self._dis = d_s
        self._dh_joint_scales = dh_joint_scales
        self._dh_joint_offsets = dh_joint_offsets

        # Forward Kinematics Constant Matrices

        # Based on Denavit-Hartenberg Convention
        # Using same nomenclature as in:
        # Modelling, Planning and Control. Bruno Siciliano, Lorenzo Sciavicco, Luigi Villani, Giuseppe Oriolo
        # page 61 - 65

        AidashtoAis_list = [self._f.identity(4, batch_shape=[1])]

        # repeated blocks

        # 1 x 1 x 3
        start_of_top_row = self._f.array([[[1., 0., 0.]]])

        # 1 x 1 x 1
        zeros = self._f.zeros((1, 1, 1))

        # 1 x 1 x 4
        bottom_row = self._f.array([[[0., 0., 0., 1.]]])

        for i in range(self._num_joints):
            # 1 x 1 x 1
            a_i = self._f.reshape(a_s[i], [1] * 3)
            cos_alpha = self._f.reshape(self._f.cos(alpha_s[i]), [1] * 3)
            sin_alpha = self._f.reshape(self._f.sin(alpha_s[i]), [1] * 3)

            # 1 x 1 x 4
            top_row = self._f.concatenate((start_of_top_row, a_i), -1)
            top_middle_row = self._f.concatenate((zeros, cos_alpha, -sin_alpha, zeros), -1)
            bottom_middle_row = self._f.concatenate((zeros, sin_alpha, cos_alpha, zeros), -1)

            # 1 x 4 x 4
            AidashtoAi = self._f.concatenate((top_row, top_middle_row, bottom_middle_row, bottom_row), 1)

            # list
            AidashtoAis_list.append(AidashtoAi)

        # NJ x 4 x 4
        self._AidashtoAis = self._f.concatenate(AidashtoAis_list, 0)

        # Constant Jacobian Params

        # Using same nomenclature as in:
        # Modelling, Planning and Control. Bruno Siciliano, Lorenzo Sciavicco, Luigi Villani, Giuseppe Oriolo
        # page 111 - 113

        # 1 x 3
        self._z0 = self._f.array([[0],
                                  [0],
                                  [1]])

        # 1 x 4
        self._p0hat = self._f.array([[0],
                                     [0],
                                     [0],
                                     [1]])

        # link lengths

        # NJ
        self._link_lengths = (a_s ** 2 + d_s ** 2) ** 0.5

    # Public Manipulator Kinematics Functions #

    # Public Methods #
    # ---------------#

    # Link poses #

    def compute_link_matrices(self, joint_angles, link_num, batch_shape=None):
        """
        Compute homogeneous transformation matrices relative to base frame, up to link_num of links.

        :param joint_angles: Joint angles of the robot *[batch_shape,num_joints]*
        :type joint_angles: array
        :param link_num: Link number for which to compute matrices up to
        :type link_num: int
        :param batch_shape: Shape of batch. Inferred from inputs if None.
        :type batch_shape: sequence of ints, optional
        :return: The link_num matrices, up the link_num *[batch_shape,link_num,4,4]*
        """

        if batch_shape is None:
            batch_shape = joint_angles.shape[:-1]
        batch_shape = list(batch_shape)
        num_batch_dims = len(batch_shape)

        # BS x 1 x NJ
        dh_joint_angles = self._f.expand_dims(joint_angles * self._dh_joint_scales - self._dh_joint_offsets, -2)

        # BS x 1 x 4 x 4
        A00 = self._f.identity(4, batch_shape=batch_shape + [1])

        Aitoip1dashs = list()
        Aiip1s = list()
        A0is = [A00]

        # repeated blocks

        # BS x 1 x NJ
        dis = self._f.tile(self._f.reshape(self._dis, [1] * num_batch_dims + [1, self._num_joints]),
                           batch_shape + [1, 1])

        # BS x 1 x 4
        bottom_row = self._f.tile(
            self._f.reshape(self._f.array([0., 0., 0., 1.]), [1] * num_batch_dims + [1, 4]),
            batch_shape + [1, 1])

        # BS x 1 x 3
        start_of_bottom_middle = self._f.tile(
            self._f.reshape(self._f.array([0., 0., 1.]), [1] * num_batch_dims + [1, 3]),
            batch_shape + [1, 1])

        # BS x 1 x 2
        zeros = self._f.zeros(batch_shape + [1, 2])

        for i in range(self._num_joints):

            # BS x 1 x 4
            top_row = self._f.concatenate((self._f.cos(dh_joint_angles[..., i:i + 1]),
                                           -self._f.sin(dh_joint_angles[..., i:i + 1]), zeros), -1)
            top_middle_row = self._f.concatenate((self._f.sin(dh_joint_angles[..., i:i + 1]),
                                                  self._f.cos(dh_joint_angles[..., i:i + 1]), zeros), -1)
            bottom_middle_row = self._f.concatenate((start_of_bottom_middle, dis[..., i:i + 1]), -1)

            # BS x 4 x 4
            Aitoip1dash = self._f.concatenate((top_row, top_middle_row, bottom_middle_row, bottom_row), -2)

            # (BSx4) x 4
            Aitoip1dash_flat = self._f.reshape(Aitoip1dash, (-1, 4))

            # (BSx4) x 4
            Aiip1_flat = self._f.matmul(Aitoip1dash_flat, self._AidashtoAis[i + 1])

            # BS x 4 x 4
            Aiip1 = self._f.reshape(Aiip1_flat, batch_shape + [4, 4])

            # BS x 4 x 4
            A0ip1 = self._f.matmul(A0is[-1][..., 0, :, :], Aiip1)

            # append term to lists
            Aitoip1dashs.append(Aitoip1dash)
            Aiip1s.append(Aiip1)
            A0is.append(self._f.expand_dims(A0ip1, -3))

            if i + 1 == link_num:
                # BS x LN x 4 x 4
                return self._f.concatenate(A0is, -3)

        raise Exception('wrong parameter entered for link_num, please enter integer from 1-' + str(self._num_joints))

    def compute_link_poses(self, joint_angles, link_num, batch_shape=None):
        """
        Compute rotation vector poses for link_num of links, starting from link 0.

        :param joint_angles: Joint angles of the robot *[batch_shape,num_joints]*
        :type joint_angles: array
        :param link_num: Link number for which to compute poses up to
        :type link_num: int
        :param batch_shape: Shape of batch. Inferred from inputs if None.
        :type batch_shape: sequence of ints, optional
        :return: The link_num poses, up the link_num *[batch_shape,link_num,6]*
        """

        if batch_shape is None:
            batch_shape = joint_angles.shape[:-1]
        batch_shape = list(batch_shape)

        # BS x LN x 4 x 4
        link_matrices = self.compute_link_matrices(joint_angles, link_num, batch_shape)

        # BS x LN x 6
        return _ivy_mech.mat_pose_to_rot_vec_pose(link_matrices[..., 0:3, :], f=self._f)

    # Link sampling #

    def sample_links(self, joint_angles, link_num=None, samples_per_metre=25, batch_shape=None):
        """
        Sample links of the robot at uniformly distributed cartesian positions.

        :param joint_angles: Joint angles of the robot *[batch_shape,num_joints]*
        :type joint_angles: array
        :param link_num: Link number for which to compute matrices up to. Default is the last link.
        :type link_num: int, optional
        :param samples_per_metre: Number of samples per metre of robot link
        :type samples_per_metre: int
        :param batch_shape: Shape of batch. Inferred from inputs if None.
        :type batch_shape: sequence of ints, optional
        :return: The sampled link cartesian positions *[batch_shape,total_sampling_chain_length,3]*
        """

        if link_num is None:
            link_num = self._num_joints
        if batch_shape is None:
            batch_shape = joint_angles.shape[:-1]
        batch_shape = list(batch_shape)
        num_batch_dims = len(batch_shape)
        batch_dims_for_trans = list(range(num_batch_dims))

        # BS x NJ x 4 x 4
        link_matrices = self.compute_link_matrices(joint_angles, link_num, batch_shape)

        # BS x LN+1 x 3
        link_positions = link_matrices[..., 0:3, -1]

        # BS x LN x 3
        segment_starts = link_positions[..., :-1, :]
        segment_ends = link_positions[..., 1:, :]

        # LN
        segment_sizes = self._f.cast(self._f.ceil(
            self._link_lengths[0:link_num] * samples_per_metre), 'int32')

        # list of segments
        segments_list = list()

        for link_idx in range(link_num):

            segment_size = segment_sizes[link_idx]

            # BS x 1 x 3
            segment_start = segment_starts[..., link_idx:link_idx + 1, :]
            segment_end = segment_ends[..., link_idx:link_idx + 1, :]

            # BS x segment_size x 3
            segment = self._f.linspace(segment_start, segment_end, segment_size, axis=-2)[..., 0, :, :]
            if link_idx == link_num - 1 or segment_size == 1:
                segments_list.append(segment)
            else:
                segments_list.append(segment[..., :-1, :])

        # BS x total_robot_chain_length x 3
        all_segments = self._f.concatenate(segments_list, -2)

        # BS x total_robot_chain_length x 4
        all_segments_homo = _ivy_mech.make_coordinates_homogeneous(all_segments, f=self._f)

        # 4 x BSxtotal_robot_chain_length
        all_segments_homo_trans = self._f.reshape(self._f.transpose(
            all_segments_homo, [num_batch_dims + 1] + batch_dims_for_trans + [num_batch_dims]), (4, -1))

        # 3 x BSxtotal_robot_chain_length
        transformed_trans = self._f.matmul(self._base_inv_ext_mat[..., 0:3, :], all_segments_homo_trans)

        # BS x total_robot_chain_length x 3
        return self._f.transpose(self._f.reshape(
            transformed_trans, [3] + batch_shape + [-1]),
            [i+1 for i in batch_dims_for_trans] + [num_batch_dims+1] + [0])


class MicoManipulator(Manipulator):

    def __init__(self, f, base_inv_ext_mat=None):
        """
        Initialize Kinova Mico robot manipulator instance.
        Denavit–Hartenberg parameters inferred from KINOVA_MICO_Robotic_arm_user_guide.pdf
        Joint scales and offsets inferred from JACO²-6DOF-Advanced-Specification-Guide.pdf
        Both of these PDFs are included in this module for reference

        :param f: Machine learning framework
        :type f: ml_framework
        :param base_inv_mat: Inverse extrinsic matrix of the robot base *[3,4]*
        :type base_inv_mat: array, optional
        """

        # framework
        self._f = f

        # length params
        # KINOVA_MICO_Robotic_arm_user_guide.pdf
        # page 50

        d1 = 0.2755
        d2 = 0.29
        d3 = 0.1233
        d4 = 0.0741
        d5 = 0.0741
        d6 = 0.16
        e2 = 0.007

        # alternate params
        # KINOVA_MICO_Robotic_arm_user_guide.pdf
        # page 53

        aa = 30 * _math.pi / 180
        sa = _math.sin(aa)
        s2a = _math.sin(2 * aa)
        d4b = d3 + sa / s2a * d4
        d5b = sa / s2a * d4 + sa / s2a * d5
        d6b = sa / s2a * d5 + d6

        # dh params
        # KINOVA_MICO_Robotic_arm_user_guide.pdf
        # page 55

        a_s = self._f.array([0, d2, 0, 0, 0, 0])
        d_s = self._f.array([d1, 0, -e2, -d4b, -d5b, -d6b])
        alpha_s = self._f.array([_math.pi / 2, _math.pi, _math.pi / 2, 2 * aa, 2 * aa, _math.pi])

        # dh joint angles convention based on:
        # JACO²-6DOF-Advanced-Specification-Guide.pdf
        # (Unable to find Mico version, but Jaco convention is the same)
        # page 10

        dh_joint_scales = self._f.array([-1., 1., 1., 1., 1., 1.])
        dh_joint_offsets = self._f.array([0., _math.pi / 2, -_math.pi / 2, 0., _math.pi, -_math.pi / 2])

        # call constructor
        super().__init__(a_s, d_s, alpha_s, dh_joint_scales, dh_joint_offsets, base_inv_ext_mat, f)


class PandaManipulator(Manipulator):

    def __init__(self, f, base_inv_ext_mat=None):
        """
        Initialize FRANKA EMIKA Panda robot manipulator instance.
        Denavit–Hartenberg parameters inferred from FRANKA EMIKA online API.
        Screenshot included in this module for reference.

        :param f: Machine learning framework
        :type f: ml_framework
        :param base_inv_mat: Inverse extrinsic matrix of the robot base *[3,4]*
        :type base_inv_mat: array, optional
        """

        # framework
        self._f = f

        # dh params
        # panda_DH_params.png
        # taken from https://frankaemika.github.io/docs/control_parameters.html

        a_s = self._f.array([0., 0., 0.0825, -0.0825, 0., 0.088, 0.])
        d_s = self._f.array([0.333, 0., 0.316, 0., 0.384, 0., 0.107])
        alpha_s = self._f.array(
            [-_math.pi / 2, _math.pi / 2, _math.pi / 2, -_math.pi / 2, _math.pi / 2, _math.pi / 2, 0.])
        dh_joint_scales = self._f.ones((7,))
        dh_joint_offsets = self._f.zeros((7,))

        super().__init__(a_s, d_s, alpha_s, dh_joint_scales, dh_joint_offsets, base_inv_ext_mat, f)

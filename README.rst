.. raw:: html

    <p align="center">
        <img width="75%" style="display: block;" src='docs/partial_source/logos/logo.png'>
    </p>

.. raw:: html

    <br/>
    <a href="https://pypi.org/project/ivy-robot">
        <img style="float: left; padding-right: 4px; padding-bottom: 4px;" src="https://badge.fury.io/py/ivy-robot.svg">
    </a>
    <a href="https://github.com/ivy-dl/robot/actions?query=workflow%3Adocs">
        <img style="float: left; padding-right: 4px; padding-bottom: 4px;" src="https://img.shields.io/github/workflow/status/ivy-dl/robot/docs?label=docs">
    </a>
    <a href="https://github.com/ivy-dl/robot/actions?query=workflow%3Anightly-tests">
        <img style="float: left; padding-right: 4px; padding-bottom: 4px;" src="https://img.shields.io/github/workflow/status/ivy-dl/robot/nightly-tests?label=tests">
    </a>
    <a href="https://discord.gg/EN9YS3QW8w">
        <img style="float: left; padding-right: 4px; padding-bottom: 4px;" src="https://img.shields.io/discord/799879767196958751?color=blue&label=%20&logo=discord&logoColor=white">
    </a>
    <br clear="all" />

**Functions and classes for gradient-based robot motion planning, written in Ivy.**

.. raw:: html

    <div style="display: block;">
        <img width="4%" style="float: left;" src="https://raw.githubusercontent.com/ivy-dl/ivy-dl.github.io/master/img/externally_linked/logos/supported/empty.png">
        <a href="https://jax.readthedocs.io">
            <img width="12%" style="float: left;" src="https://raw.githubusercontent.com/ivy-dl/ivy-dl.github.io/master/img/externally_linked/logos/supported/jax_logo.png">
        </a>
        <img width="6.5%" style="float: left;" src="https://raw.githubusercontent.com/ivy-dl/ivy-dl.github.io/master/img/externally_linked/logos/supported/empty.png">
        <a href="https://www.tensorflow.org">
            <img width="12%" style="float: left;" src="https://raw.githubusercontent.com/ivy-dl/ivy-dl.github.io/master/img/externally_linked/logos/supported/tensorflow_logo.png">
        </a>
        <img width="6.5%" style="float: left;" src="https://raw.githubusercontent.com/ivy-dl/ivy-dl.github.io/master/img/externally_linked/logos/supported/empty.png">
        <a href="https://pytorch.org">
            <img width="12%" style="float: left;" src="https://raw.githubusercontent.com/ivy-dl/ivy-dl.github.io/master/img/externally_linked/logos/supported/pytorch_logo.png">
        </a>
        <img width="6.5%" style="float: left;" src="https://raw.githubusercontent.com/ivy-dl/ivy-dl.github.io/master/img/externally_linked/logos/supported/empty.png">
        <a href="https://mxnet.apache.org">
            <img width="12%" style="float: left;" src="https://raw.githubusercontent.com/ivy-dl/ivy-dl.github.io/master/img/externally_linked/logos/supported/mxnet_logo.png">
        </a>
        <img width="6.5%" style="float: left;" src="https://raw.githubusercontent.com/ivy-dl/ivy-dl.github.io/master/img/externally_linked/logos/supported/empty.png">
        <a href="https://numpy.org">
            <img width="12%" style="float: left;" src="https://raw.githubusercontent.com/ivy-dl/ivy-dl.github.io/master/img/externally_linked/logos/supported/numpy_logo.png">
        </a>
    </div>

Contents
--------

* `Overview`_
* `Run Through`_
* `Interactive Demos`_
* `Get Involed`_

Overview
--------

.. _docs: https://ivy-dl.org/robot

**What is Ivy Robot?**

Ivy robot provides functions and classes for gradient-based motion planning and trajectory optimization.
Classes are provided both for mobile robots and robot manipulators.  Check out the docs_ for more info!

The library is built on top of the Ivy deep learning framework.
This means all functions and classes simultaneously support:
Jax, Tensorflow, PyTorch, MXNet, and Numpy.

**Ivy Libraries**

There are a host of derived libraries written in Ivy, in the areas of mechanics, 3D vision, robotics,
differentiable memory, and differentiable gym environments. Click on the icons below for their respective github pages.

.. raw:: html

    <div style="display: block;">
        <img width="9%" style="float: left;" src="https://raw.githubusercontent.com/ivy-dl/ivy-dl.github.io/master/img/externally_linked/logos/empty.png">
        <a href="https://github.com/ivy-dl/mech">
            <img width="15%" style="float: left;" src="https://raw.githubusercontent.com/ivy-dl/ivy-dl.github.io/master/img/externally_linked/logos/ivy_mech.png">
        </a>
        <img width="7%" style="float: left;" src="https://raw.githubusercontent.com/ivy-dl/ivy-dl.github.io/master/img/externally_linked/logos/empty.png">
        <a href="https://github.com/ivy-dl/vision">
            <img width="15%" style="float: left;" src="https://raw.githubusercontent.com/ivy-dl/ivy-dl.github.io/master/img/externally_linked/logos/ivy_vision.png">
        </a>
        <img width="7%" style="float: left;" src="https://raw.githubusercontent.com/ivy-dl/ivy-dl.github.io/master/img/externally_linked/logos/empty.png">
        <a href="https://github.com/ivy-dl/robot">
            <img width="15%" style="float: left;" src="https://raw.githubusercontent.com/ivy-dl/ivy-dl.github.io/master/img/externally_linked/logos/ivy_robot.png">
        </a>
        <img width="7%" style="float: left;" src="https://raw.githubusercontent.com/ivy-dl/ivy-dl.github.io/master/img/externally_linked/logos/empty.png">
        <a href="https://github.com/ivy-dl/gym">
            <img width="15%" style="float: left;" src="https://raw.githubusercontent.com/ivy-dl/ivy-dl.github.io/master/img/externally_linked/logos/ivy_gym.png">
        </a>

        <br clear="all" />

        <img width="10%" style="float: left;" src="https://raw.githubusercontent.com/ivy-dl/ivy-dl.github.io/master/img/externally_linked/logos/empty.png">
        <a href="https://pypi.org/project/ivy-mech">
            <img width="13%" style="float: left;" src="https://badge.fury.io/py/ivy-mech.svg">
        </a>
        <img width="9%" style="float: left;" src="https://raw.githubusercontent.com/ivy-dl/ivy-dl.github.io/master/img/externally_linked/logos/empty.png">
        <a href="https://pypi.org/project/ivy-vision">
            <img width="13%" style="float: left;" src="https://badge.fury.io/py/ivy-vision.svg">
        </a>
        <img width="9%" style="float: left;" src="https://raw.githubusercontent.com/ivy-dl/ivy-dl.github.io/master/img/externally_linked/logos/empty.png">
        <a href="https://pypi.org/project/ivy-robot">
            <img width="13%" style="float: left;" src="https://badge.fury.io/py/ivy-robot.svg">
        </a>
        <img width="9%" style="float: left;" src="https://raw.githubusercontent.com/ivy-dl/ivy-dl.github.io/master/img/externally_linked/logos/empty.png">
        <a href="https://pypi.org/project/ivy-gym">
            <img width="13%" style="float: left;" src="https://badge.fury.io/py/ivy-gym.svg">
        </a>

        <br clear="all" />

        <img width="12%" style="float: left;" src="https://raw.githubusercontent.com/ivy-dl/ivy-dl.github.io/master/img/externally_linked/logos/empty.png">
        <a href="https://github.com/ivy-dl/mech/actions?query=workflow%3Anightly-tests">
            <img width="9%" style="float: left;" src="https://img.shields.io/github/workflow/status/ivy-dl/mech/nightly-tests?label=tests">
        </a>
        <img width="13%" style="float: left;" src="https://raw.githubusercontent.com/ivy-dl/ivy-dl.github.io/master/img/externally_linked/logos/empty.png">
        <a href="https://github.com/ivy-dl/vision/actions?query=workflow%3Anightly-tests">
            <img width="9%" style="float: left;" src="https://img.shields.io/github/workflow/status/ivy-dl/vision/nightly-tests?label=tests">
        </a>
        <img width="13%" style="float: left;" src="https://raw.githubusercontent.com/ivy-dl/ivy-dl.github.io/master/img/externally_linked/logos/empty.png">
        <a href="https://github.com/ivy-dl/robot/actions?query=workflow%3Anightly-tests">
            <img width="9%" style="float: left;" src="https://img.shields.io/github/workflow/status/ivy-dl/robot/nightly-tests?label=tests">
        </a>
        <img width="13%" style="float: left;" src="https://raw.githubusercontent.com/ivy-dl/ivy-dl.github.io/master/img/externally_linked/logos/empty.png">
        <a href="https://github.com/ivy-dl/gym/actions?query=workflow%3Anightly-tests">
            <img width="9%" style="float: left;" src="https://img.shields.io/github/workflow/status/ivy-dl/gym/nightly-tests?label=tests">
        </a>

        <br clear="all" />

        <img width="20%" style="float: left;" src="https://raw.githubusercontent.com/ivy-dl/ivy-dl.github.io/master/img/externally_linked/logos/empty.png">
        <a href="https://github.com/ivy-dl/memory">
            <img width="15%" style="float: left;" src="https://raw.githubusercontent.com/ivy-dl/ivy-dl.github.io/master/img/externally_linked/logos/ivy_memory.png">
        </a>
        <img width="7%" style="float: left;" src="https://raw.githubusercontent.com/ivy-dl/ivy-dl.github.io/master/img/externally_linked/logos/empty.png">
        <a href="https://github.com/ivy-dl/builder">
            <img width="15%" style="float: left;" src="https://raw.githubusercontent.com/ivy-dl/ivy-dl.github.io/master/img/externally_linked/logos/ivy_builder.png">
        </a>
        <img width="7%" style="float: left;" src="https://raw.githubusercontent.com/ivy-dl/ivy-dl.github.io/master/img/externally_linked/logos/empty.png">
        <a href="https://github.com/ivy-dl/models">
            <img width="15%" style="float: left;" src="https://raw.githubusercontent.com/ivy-dl/ivy-dl.github.io/master/img/externally_linked/logos/ivy_models.png">
        </a>

        <br clear="all" />

        <img width="21%" style="float: left;" src="https://raw.githubusercontent.com/ivy-dl/ivy-dl.github.io/master/img/externally_linked/logos/empty.png">
        <a href="https://pypi.org/project/ivy-memory">
            <img width="13%" style="float: left;" src="https://badge.fury.io/py/ivy-memory.svg">
        </a>
        <img width="9%" style="float: left;" src="https://raw.githubusercontent.com/ivy-dl/ivy-dl.github.io/master/img/externally_linked/logos/empty.png">
        <a href="https://pypi.org/project/ivy-builder">
            <img width="13%" style="float: left;" src="https://badge.fury.io/py/ivy-builder.svg">
        </a>
        <img width="9%" style="float: left;" src="https://raw.githubusercontent.com/ivy-dl/ivy-dl.github.io/master/img/externally_linked/logos/empty.png">
        <a href="https://pypi.org/project/ivy-models">
            <img width="13%" style="float: left;" src="https://badge.fury.io/py/ivy-models.svg">
        </a>

        <br clear="all" />

        <img width="23%" style="float: left;" src="https://raw.githubusercontent.com/ivy-dl/ivy-dl.github.io/master/img/externally_linked/logos/empty.png">
        <a href="https://github.com/ivy-dl/memory/actions?query=workflow%3Anightly-tests">
            <img width="9%" style="float: left;" src="https://img.shields.io/github/workflow/status/ivy-dl/memory/nightly-tests?label=tests">
        </a>
        <img width="13%" style="float: left;" src="https://raw.githubusercontent.com/ivy-dl/ivy-dl.github.io/master/img/externally_linked/logos/empty.png">
        <a href="https://github.com/ivy-dl/builder/actions?query=workflow%3Anightly-tests">
            <img width="9%" style="float: left;" src="https://img.shields.io/github/workflow/status/ivy-dl/builder/nightly-tests?label=tests">
        </a>
        <img width="13%" style="float: left;" src="https://raw.githubusercontent.com/ivy-dl/ivy-dl.github.io/master/img/externally_linked/logos/empty.png">
        <a href="https://github.com/ivy-dl/models/actions?query=workflow%3Anightly-tests">
            <img width="9%" style="float: left;" src="https://img.shields.io/github/workflow/status/ivy-dl/models/nightly-tests?label=tests">
        </a>

        <br clear="all" />

    </div>
    <br clear="all" />

**Quick Start**

Ivy robot can be installed like so: ``pip install ivy-robot``

.. _demos: https://github.com/ivy-dl/robot/tree/master/ivy_robot_demos
.. _interactive: https://github.com/ivy-dl/robot/tree/master/ivy_robot_demos/interactive

To quickly see the different aspects of the library, we suggest you check out the demos_!
We suggest you start by running the script ``run_through.py``,
and read the "Run Through" section below which explains this script.

For more interactive demos, we suggest you run either
``drone_spline_planning.py`` or ``manipulator_spline_planning.py`` in the interactive_ demos folder.

Run Through
-----------

We run through some of the different parts of the library via a simple ongoing example script.
The full script is available in the demos_ folder, as file ``run_through.py``.
First, we select a random backend framework to use for the examples, from the options
``ivy.jax``, ``ivy.tensorflow``, ``ivy.torch``, ``ivy.mxnet`` or ``ivy.numpy``,
and use this to set the ivy backend framework.

.. code-block:: python

    import ivy
    from ivy_demo_utils.framework_utils import choose_random_framework
    ivy.set_framework(choose_random_framework())

**Spline Planning**

We now show how a spline path can be generated from a set of spline anchor points,
using the method ``ivy_robot.sample_spline_path``.
In this example, we generate a spline path representing full 6DOF motion from a starting pose to a target pose.
However, for simplicitly we fix the z translation and 3DOF rotation to zeros in this case.

.. code-block:: python

    # config
    num_free_anchors = 3
    num_samples = 100
    constant_rot_vec = ivy.array([[0., 0., 0.]])
    constant_z = ivy.array([[0.]])

    # xy positions

    # 1 x 2
    start_xy = ivy.array([[0., 0.]])
    target_xy = ivy.array([[1., 1.]])

    # 1 x 2
    anchor1_xy = ivy.array([[0.6, 0.2]])
    anchor2_xy = ivy.array([[0.5, 0.5]])
    anchor3_xy = ivy.array([[0.4, 0.8]])

    # as 6DOF poses

    # 1 x 6
    start_pose = ivy.concatenate((start_xy, constant_z, constant_rot_vec), -1)
    anchor1_pose = ivy.concatenate((anchor1_xy, constant_z, constant_rot_vec), -1)
    anchor2_pose = ivy.concatenate((anchor2_xy, constant_z, constant_rot_vec), -1)
    anchor3_pose = ivy.concatenate((anchor3_xy, constant_z, constant_rot_vec), -1)
    target_pose = ivy.concatenate((target_xy, constant_z, constant_rot_vec), -1)

    num_anchors = num_free_anchors + 2

    # num_anchors x 6
    anchor_poses = ivy.concatenate((start_pose, anchor1_pose, anchor2_pose, anchor3_pose, target_pose), 0)

    # uniform sampling for spline

    # num_anchors x 1
    anchor_points = ivy.expand_dims(ivy.linspace(0., 1., num_anchors), -1)

    # num_samples x 1
    query_points = ivy.expand_dims(ivy.linspace(0., 1., num_samples), -1)

    # interpolated spline poses

    # num_samples x 6
    interpolated_poses = ivy_robot.sample_spline_path(anchor_points, anchor_poses, query_points)

    # xy motion

    # num_samples x 2
    anchor_xy_positions = anchor_poses[..., 0:2]

    # num_samples x 2
    interpolated_xy_positions = interpolated_poses[..., 0:2]

The interpolated xy positions and anchor positions from the path are shown below in the x-y plane.

.. raw:: html

    <p align="center">
        <img width="75%" style="display: block;" src='docs/partial_source/images/interpolated_drone_poses.png'>
    </p>

**Rigid Mobile Class**

We now introduce the ``RigidMobile`` robot class,
which can be used to represent rigid jointless robots which are able to move freely.
In this case, we consider the case of a drone executing 6DOF motion in a scene.

The body of the drone is specified by a number of relative body points.
In this case, we represent the drone with 5 points: one in the centre, and one in each of the four corners.

We assume the same target position in the x-y plane as before,
but this time with a self-rotation of 180 degrees about the z-axis.

.. code-block:: python

    # drone relative body points
    rel_body_points = ivy.array([[0., 0., 0.],
                               [-0.1, -0.1, 0.],
                               [-0.1, 0.1, 0.],
                               [0.1, -0.1, 0.],
                               [0.1, 0.1, 0.]])

    # create drone as ivy rigid mobile robot
    drone = RigidMobile(rel_body_points)

    # rotatin vectors

    # 1 x 3
    start_rot_vec = ivy.array([[0., 0., 0.]])
    target_rot_vec = ivy.array([[0., 0., np.pi]])

    # 1 x 3
    anchor1_rot_vec = ivy.array([[0., 0., np.pi/4]])
    anchor2_rot_vec = ivy.array([[0., 0., 2*np.pi/4]])
    anchor3_rot_vec = ivy.array([[0., 0., 3*np.pi/4]])

    # as 6DOF poses

    # 1 x 6
    start_pose = ivy.concatenate((start_xy, constant_z, start_rot_vec), -1)
    anchor1_pose = ivy.concatenate((anchor1_xy, constant_z, anchor1_rot_vec), -1)
    anchor2_pose = ivy.concatenate((anchor2_xy, constant_z, anchor2_rot_vec), -1)
    anchor3_pose = ivy.concatenate((anchor3_xy, constant_z, anchor3_rot_vec), -1)
    target_pose = ivy.concatenate((target_xy, constant_z, target_rot_vec), -1)

    # num_anchors x 6
    anchor_poses = ivy.concatenate((start_pose, anchor1_pose, anchor2_pose, anchor3_pose, target_pose), 0)

    # interpolated spline poses

    # num_samples x 6
    interpolated_poses = ivy_robot.sample_spline_path(anchor_points, anchor_poses, query_points)

    # as matrices

    # num_anchors x 3 x 4
    anchor_matrices = ivy_mech.rot_vec_pose_to_mat_pose(anchor_poses)

    # num_samples x 3 x 4
    interpolated_matrices = ivy_mech.rot_vec_pose_to_mat_pose(interpolated_poses)

    # sample drone body

    # num_anchors x num_body_points x 3
    anchor_body_points = drone.sample_body(anchor_matrices)

    # num_samples x num_body_points x 3
    interpolated_body_points = drone.sample_body(interpolated_matrices)

The sampled drone body xy positions during motion are shown below in the x-y plane.
By tracing the body points for each of the four corners of the drone,
we can see how the drone performs the 180 degree self-rotation about the z-axis during the motion.

.. raw:: html

    <p align="center">
        <img width="75%" style="display: block;" src='docs/partial_source/images/sampled_drone_body_positions.png'>
    </p>

**Manipulator Class**

We now introduce the robot Manipulator class,
which can be used to represent arbitrary robot manipulators.
In this case, we consider the case of very simple 2-link manipulator,
which is constrained to move in the x-y plane.

The manipulator is specified by the Denavit–Hartenberg parameters,
as outlined in the newly derived class below.
We assume a manipulator with two 0.5m links,
where a configuration with both joints angles at 0 degrees represents a upright link configuration.
We specify a new set of target joint angles which corresponds with
a forward reaching motion in the positive x direction.

.. code-block:: python

    class SimpleManipulator(Manipulator):

        def __init__(self, base_inv_ext_mat=None):
            a_s = ivy.array([0.5, 0.5])
            d_s = ivy.array([0., 0.])
            alpha_s = ivy.array([0., 0.])
            dh_joint_scales = ivy.ones((2,))
            dh_joint_offsets = ivy.array([-np.pi/2, 0.])
            super().__init__(a_s, d_s, alpha_s, dh_joint_scales, dh_joint_offsets, base_inv_ext_mat)

    # create manipulator as ivy manipulator
    manipulator = SimpleManipulator()

    # joint angles

    # 1 x 2
    start_joint_angles = ivy.array([[0., 0.]])
    target_joint_angles = ivy.array([[-np.pi/4, -np.pi/4]])

    # 1 x 2
    anchor1_joint_angles = -ivy.array([[0.2, 0.6]])*np.pi/4
    anchor2_joint_angles = -ivy.array([[0.5, 0.5]])*np.pi/4
    anchor3_joint_angles = -ivy.array([[0.8, 0.4]])*np.pi/4

    # num_anchors x 2
    anchor_joint_angles = ivy.concatenate(
        (start_joint_angles, anchor1_joint_angles, anchor2_joint_angles, anchor3_joint_angles,
         target_joint_angles), 0)

    # interpolated joint angles

    # num_anchors x 2
    interpolated_joint_angles = ivy_robot.sample_spline_path(anchor_points, anchor_joint_angles, query_points)

The interpolated joint angles are presented below.

.. raw:: html

    <p align="center">
        <img width="75%" style="display: block;" src='docs/partial_source/images/interpolated_manipulator_joint_angles.png'>
    </p>

In a similar fashion to how the drone body was sampled in the previous example,
we next use these interpolated joint angles to sample the link positions for the manipulator.

.. code-block:: python

    # sample links

    # num_anchors x num_link_points x 3
    anchor_link_points = manipulator.sample_links(anchor_joint_angles, samples_per_metre=5)

    # num_anchors x num_link_points x 3
    interpolated_link_points = manipulator.sample_links(interpolated_joint_angles, samples_per_metre=5)

we show the sampled link positions during the course of the forward reaching motion in the x-y plane below.

.. raw:: html

    <p align="center">
        <img width="75%" style="display: block;" src='docs/partial_source/images/sampled_manipulator_links.png'>
    </p>

Interactive Demos
-----------------

The main benefit of the library is not simply the ability to sample paths, but to optimize these paths using gradients.
For exmaple, the body or link sample positions can be used to query the signed distance function (SDF) of a 3D scene in batch.
Then, assuming the spline anchor points to be free variables,
the gradients of the mean sampled SDF and a path length metric can be computed with respect to the anchor points.
The anhcor points can then be incrementally updated using gradient descent on this loss function.

We provide two further demo scripts which outline this gradient-based path optimization in a 3D scene.
Rather than presenting the code here, we show visualizations of the demos.
The scripts for these demos can be found in the interactive_ demos folder.

**RigidMobile Planning**

The first demo uses the ``RigidMobile`` class to optimzie the motion of a drone to a target pose,
making use of functions ``ivy_robot.sample_spline_path`` and ``ivy_robot.RigidMobile.sample_body``.

.. raw:: html

    <p align="center">
        <img width="75%" style="display: block;" src='https://github.com/ivy-dl/ivy-dl.github.io/blob/master/img/externally_linked/ivy_robot/demo_a.gif?raw=true'>
    </p>

**Manipulator Planning**

The second demo uses the ``MicoManipulator`` class, derived from ``Manipulator``,
to optimzie the motion of a mico robot manipulator to a set of target joint angles,
making use of functions ``ivy_robot.sample_spline_path`` and ``ivy_robot.Manipulator.sample_links``.

.. raw:: html

    <p align="center">
        <img width="75%" style="display: block;" src='https://github.com/ivy-dl/ivy-dl.github.io/blob/master/img/externally_linked/ivy_robot/demo_b.gif?raw=true'>
    </p>

Get Involed
-----------

We hope the functions in this library are useful to a wide range of deep learning developers.
However, there are many more areas of gradient-based motion planning and broader robotics
which could be covered by this library.

If there are any particular robotics functions you feel are missing,
and your needs are not met by the functions currently on offer,
then we are very happy to accept pull requests!

We look forward to working with the community on expanding and improving the Ivy robot library.

Citation
--------

::

    @article{lenton2021ivy,
      title={Ivy: Templated Deep Learning for Inter-Framework Portability},
      author={Lenton, Daniel and Pardo, Fabio and Falck, Fabian and James, Stephen and Clark, Ronald},
      journal={arXiv preprint arXiv:2102.02886},
      year={2021}
    }
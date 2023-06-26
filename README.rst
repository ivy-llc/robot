.. image:: https://github.com/unifyai/unifyai.github.io/blob/master/img/externally_linked/logo.png?raw=true#gh-light-mode-only
   :width: 100%
   :class: only-light

.. image:: https://github.com/unifyai/unifyai.github.io/blob/master/img/externally_linked/logo_dark.png?raw=true#gh-dark-mode-only
   :width: 100%
   :class: only-dark

.. raw:: html

    <br/>
    <a href="https://pypi.org/project/ivy-robot">
        <img class="dark-light" style="float: left; padding-right: 4px; padding-bottom: 4px;" src="https://badge.fury.io/py/ivy-robot.svg">
    </a>
    <a href="https://github.com/unifyai/robot/actions?query=workflow%3Adocs">
        <img class="dark-light" style="float: left; padding-right: 4px; padding-bottom: 4px;" src="https://github.com/unifyai/robot/actions/workflows/docs.yml/badge.svg">
    </a>
    <a href="https://github.com/unifyai/robot/actions?query=workflow%3Anightly-tests">
        <img class="dark-light" style="float: left; padding-right: 4px; padding-bottom: 4px;" src="https://github.com/unifyai/robot/actions/workflows/nightly-tests.yml/badge.svg">
    </a>
    <a href="https://discord.gg/G4aR9Q7DTN">
        <img class="dark-light" style="float: left; padding-right: 4px; padding-bottom: 4px;" src="https://img.shields.io/discord/799879767196958751?color=blue&label=%20&logo=discord&logoColor=white">
    </a>
    <br clear="all" />

**Functions and classes for gradient-based robot motion planning, written in Ivy.**

.. raw:: html

    <div style="display: block;" align="center">
        <img class="dark-light" width="6%" style="float: left;" src="https://raw.githubusercontent.com/unifyai/unifyai.github.io/master/img/externally_linked/logos/supported/empty.png">
        <a href="https://jax.readthedocs.io">
            <img class="dark-light" width="13%" style="float: left;" src="https://raw.githubusercontent.com/unifyai/unifyai.github.io/master/img/externally_linked/logos/supported/jax_logo.png">
        </a>
        <img class="dark-light" width="12%" style="float: left;" src="https://raw.githubusercontent.com/unifyai/unifyai.github.io/master/img/externally_linked/logos/supported/empty.png">
        <a href="https://www.tensorflow.org">
            <img class="dark-light" width="13%" style="float: left;" src="https://raw.githubusercontent.com/unifyai/unifyai.github.io/master/img/externally_linked/logos/supported/tensorflow_logo.png">
        </a>
        <img class="dark-light" width="12%" style="float: left;" src="https://raw.githubusercontent.com/unifyai/unifyai.github.io/master/img/externally_linked/logos/supported/empty.png">
        <a href="https://pytorch.org">
            <img class="dark-light" width="13%" style="float: left;" src="https://raw.githubusercontent.com/unifyai/unifyai.github.io/master/img/externally_linked/logos/supported/pytorch_logo.png">
        </a>
        <img class="dark-light" width="12%" style="float: left;" src="https://raw.githubusercontent.com/unifyai/unifyai.github.io/master/img/externally_linked/logos/supported/empty.png">
        <a href="https://numpy.org">
            <img class="dark-light" width="13%" style="float: left;" src="https://raw.githubusercontent.com/unifyai/unifyai.github.io/master/img/externally_linked/logos/supported/numpy_logo.png">
        </a>
        <img class="dark-light" width="6%" style="float: left;" src="https://raw.githubusercontent.com/unifyai/unifyai.github.io/master/img/externally_linked/logos/supported/empty.png">
    </div>

Contents
--------

* `Overview`_
* `Run Through`_
* `Interactive Demos`_
* `Get Involved`_

Overview
--------

.. _docs: https://lets-unify.ai/robot

**What is Ivy Robot?**

Ivy robot provides functions and classes for gradient-based motion planning and trajectory optimization.
Classes are provided both for mobile robots and robot manipulators.  Check out the docs_ for more info!

The library is built on top of the Ivy machine learning framework.
This means all functions and classes simultaneously support:
Jax, Tensorflow, PyTorch, MXNet, and Numpy.

**Ivy Libraries**

There are a host of derived libraries written in Ivy, in the areas of mechanics, 3D vision, robotics, gym environments,
neural memory, pre-trained models + implementations, and builder tools with trainers, data loaders and more. Click on the icons below to learn more!

.. raw:: html

    <div style="display: block;">
        <a href="https://github.com/unifyai/mech">
            <picture>
                <source class="dark-light" width="15%" style="float: left; margin: 0% 5%;" media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/unifyai/unifyai.github.io/master/img/externally_linked/logos/ivy_mech_dark.png">
                <img class="dark-light" width="15%" style="float: left; margin: 0% 5%;" src="https://raw.githubusercontent.com/unifyai/unifyai.github.io/master/img/externally_linked/logos/ivy_mech.png">
            </picture>
        </a>
        <a href="https://github.com/unifyai/vision">
            <picture>
                <source class="dark-light" width="15%" style="float: left; margin: 0% 5%;" media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/unifyai/unifyai.github.io/master/img/externally_linked/logos/ivy_vision_dark.png">
                <img class="dark-light" width="15%" style="float: left; margin: 0% 5%;" src="https://raw.githubusercontent.com/unifyai/unifyai.github.io/master/img/externally_linked/logos/ivy_vision.png">
            </picture>
        </a>
        <a href="https://github.com/unifyai/robot">
            <picture>
                <source class="dark-light" width="15%" style="float: left; margin: 0% 5%;" media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/unifyai/unifyai.github.io/master/img/externally_linked/logos/ivy_robot_dark.png">
                <img class="dark-light" width="15%" style="float: left; margin: 0% 5%;" src="https://raw.githubusercontent.com/unifyai/unifyai.github.io/master/img/externally_linked/logos/ivy_robot.png">
            </picture>
        </a>
        <a href="https://github.com/unifyai/gym">
            <picture>
                <source class="dark-light" width="15%" style="float: left; margin: 0% 5%;" media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/unifyai/unifyai.github.io/master/img/externally_linked/logos/ivy_gym_dark.png">
                <img class="dark-light" width="15%" style="float: left; margin: 0% 5%;" src="https://raw.githubusercontent.com/unifyai/unifyai.github.io/master/img/externally_linked/logos/ivy_gym.png">
            </picture>
        </a>

        <br clear="all" />

        <a href="https://pypi.org/project/ivy-mech">
            <img class="dark-light" width="15%" style="float: left; margin: 0% 5%;" src="https://badge.fury.io/py/ivy-mech.svg">
        </a>
        <a href="https://pypi.org/project/ivy-vision">
            <img class="dark-light" width="15%" style="float: left; margin: 0% 5%;" src="https://badge.fury.io/py/ivy-vision.svg">
        </a>
        <a href="https://pypi.org/project/ivy-robot">
            <img class="dark-light" width="15%" style="float: left; margin: 0% 5%;" src="https://badge.fury.io/py/ivy-robot.svg">
        </a>
        <a href="https://pypi.org/project/ivy-gym">
            <img class="dark-light" width="15%" style="float: left; margin: 0% 5%;"width="15%" style="float: left; margin: 0% 5%;" src="https://badge.fury.io/py/ivy-gym.svg">
        </a>

        <br clear="all" />

        <a href="https://github.com/unifyai/mech/actions?query=workflow%3Anightly-tests">
            <img class="dark-light" width="15%" style="float: left; margin: 0% 5%;"src="https://github.com/unifyai/mech/actions/workflows/nightly-tests.yml/badge.svg">
        </a>
        <a href="https://github.com/unifyai/vision/actions?query=workflow%3Anightly-tests">
            <img class="dark-light" width="15%" style="float: left; margin: 0% 5%;" src="https://github.com/unifyai/vision/actions/workflows/nightly-tests.yml/badge.svg">
        </a>
        <a href="https://github.com/unifyai/robot/actions?query=workflow%3Anightly-tests">
            <img class="dark-light" width="15%" style="float: left; margin: 0% 5%;" src="https://github.com/unifyai/robot/actions/workflows/nightly-tests.yml/badge.svg">
        </a>
        <a href="https://github.com/unifyai/gym/actions?query=workflow%3Anightly-tests">
            <img class="dark-light" width="15%" style="float: left; margin: 0% 5%;" src="https://github.com/unifyai/gym/actions/workflows/nightly-tests.yml/badge.svg">
        </a>

        <br clear="all" />

        <a href="https://github.com/unifyai/memory">
            <picture>
                <source class="dark-light" width="15%" style="float: left; margin: 0% 5%;" media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/unifyai/unifyai.github.io/master/img/externally_linked/logos/ivy_memory_dark.png">
                <img class="dark-light" width="15%" style="float: left; margin: 0% 5%;" src="https://raw.githubusercontent.com/unifyai/unifyai.github.io/master/img/externally_linked/logos/ivy_memory.png">
            </picture>
        </a>
        <a href="https://github.com/unifyai/builder">
            <picture>
                <source class="dark-light" width="15%" style="float: left; margin: 0% 5%;" media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/unifyai/unifyai.github.io/master/img/externally_linked/logos/ivy_builder_dark.png">
                <img class="dark-light" width="15%" style="float: left; margin: 0% 5%;" src="https://raw.githubusercontent.com/unifyai/unifyai.github.io/master/img/externally_linked/logos/ivy_builder.png">
            </picture>
        </a>
        <a href="https://github.com/unifyai/models">
            <picture>
                <source class="dark-light" width="15%" style="float: left; margin: 0% 5%;" media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/unifyai/unifyai.github.io/master/img/externally_linked/logos/ivy_models_dark.png">
                <img class="dark-light" width="15%" style="float: left; margin: 0% 5%;" src="https://raw.githubusercontent.com/unifyai/unifyai.github.io/master/img/externally_linked/logos/ivy_models.png">
            </picture>
        </a>
        <a href="https://github.com/unifyai/ecosystem">
            <picture>
                <source class="dark-light" width="15%" style="float: left; margin: 0% 5%;" media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/unifyai/unifyai.github.io/master/img/externally_linked/logos/ivy_ecosystem_dark.png">
                <img class="dark-light" width="15%" style="float: left; margin: 0% 5%;" src="https://raw.githubusercontent.com/unifyai/unifyai.github.io/master/img/externally_linked/logos/ivy_ecosystem.png">
            </picture>
        </a>

        <br clear="all" />

        <a href="https://pypi.org/project/ivy-memory">
            <img class="dark-light" width="15%" style="float: left; margin: 0% 5%;" src="https://badge.fury.io/py/ivy-memory.svg">
        </a>
        <a href="https://pypi.org/project/ivy-builder">
            <img class="dark-light" width="15%" style="float: left; margin: 0% 5%;" src="https://badge.fury.io/py/ivy-builder.svg">
        </a>
        <a href="https://pypi.org/project/ivy-models">
            <img class="dark-light" width="15%" style="float: left; margin: 0% 5%;" src="https://badge.fury.io/py/ivy-models.svg">
        </a>
        <a href="https://github.com/unifyai/ecosystem/actions?query=workflow%3Adocs">
            <img class="dark-light" width="15%" style="float: left; margin: 0% 5%;" src="https://github.com/unifyai/ecosystem/actions/workflows/docs.yml/badge.svg">
        </a>

        <br clear="all" />

        <a href="https://github.com/unifyai/memory/actions?query=workflow%3Anightly-tests">
            <img class="dark-light" width="15%" style="float: left; margin: 0% 5%;" src="https://github.com/unifyai/memory/actions/workflows/nightly-tests.yml/badge.svg">
        </a>
        <a href="https://github.com/unifyai/builder/actions?query=workflow%3Anightly-tests">
            <img class="dark-light" width="15%" style="float: left; margin: 0% 5%;" src="https://github.com/unifyai/builder/actions/workflows/nightly-tests.yml/badge.svg">
        </a>
        <a href="https://github.com/unifyai/models/actions?query=workflow%3Anightly-tests">
            <img class="dark-light" width="15%" style="float: left; margin: 0% 5%;" src="https://github.com/unifyai/models/actions/workflows/nightly-tests.yml/badge.svg">
        </a>

        <br clear="all" />

    </div>
    <br clear="all" />

**Quick Start**

Ivy robot can be installed like so: ``pip install ivy-robot``

.. _demos: https://github.com/unifyai/robot/tree/master/ivy_robot_demos
.. _interactive: https://github.com/unifyai/robot/tree/master/ivy_robot_demos/interactive

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
    ivy.set_backend(ivy.choose_random_backend())

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
    start_pose = ivy.concat((start_xy, constant_z, constant_rot_vec), axis=-1)
    anchor1_pose = ivy.concat((anchor1_xy, constant_z, constant_rot_vec), axis=-1)
    anchor2_pose = ivy.concat((anchor2_xy, constant_z, constant_rot_vec), axis=-1)
    anchor3_pose = ivy.concat((anchor3_xy, constant_z, constant_rot_vec), axis=-1)
    target_pose = ivy.concat((target_xy, constant_z, constant_rot_vec), axis=-1)

    num_anchors = num_free_anchors + 2

    # num_anchors x 6
    anchor_poses = ivy.concat((start_pose, anchor1_pose, anchor2_pose, anchor3_pose, target_pose), axis=0)

    # uniform sampling for spline

    # num_anchors x 1
    anchor_points = ivy.expand_dims(ivy.linspace(0., 1., num_anchors), axis=-1)

    # num_samples x 1
    query_points = ivy.expand_dims(ivy.linspace(0., 1., num_samples), axis=-1)

    # interpolated spline poses

    # num_samples x 6
    interpolated_poses = ivy_robot.sample_spline_path(anchor_points, anchor_poses, query_points)

    # xy motion

    # num_samples x 2
    anchor_xy_positions = anchor_poses[..., 0:2]

    # num_samples x 2
    interpolated_xy_positions = interpolated_poses[..., 0:2]

The interpolated xy positions and anchor positions from the path are shown below in the x-y plane.

.. image:: https://github.com/unifyai/robot/blob/master/docs/images/interpolated_drone_poses.png?raw=true
   :width: 100%

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
    start_pose = ivy.concat((start_xy, constant_z, start_rot_vec), axis=-1)
    anchor1_pose = ivy.concat((anchor1_xy, constant_z, anchor1_rot_vec), axis=-1)
    anchor2_pose = ivy.concat((anchor2_xy, constant_z, anchor2_rot_vec), axis=-1)
    anchor3_pose = ivy.concat((anchor3_xy, constant_z, anchor3_rot_vec), axis=-1)
    target_pose = ivy.concat((target_xy, constant_z, target_rot_vec), axis=-1)

    # num_anchors x 6
    anchor_poses = ivy.concat((start_pose, anchor1_pose, anchor2_pose, anchor3_pose, target_pose), axis=0)

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

.. image:: https://github.com/unifyai/robot/blob/master/docs/images/sampled_drone_body_positions.png?raw=true
   :width: 100%

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
    anchor_joint_angles = ivy.concat(
        (start_joint_angles, anchor1_joint_angles, anchor2_joint_angles, anchor3_joint_angles,
         target_joint_angles), axis=0)

    # interpolated joint angles

    # num_anchors x 2
    interpolated_joint_angles = ivy_robot.sample_spline_path(anchor_points, anchor_joint_angles, query_points)

The interpolated joint angles are presented below.

.. image:: https://github.com/unifyai/robot/blob/master/docs/images/interpolated_manipulator_joint_angles.png?raw=true
   :width: 100%

In a similar fashion to how the drone body was sampled in the previous example,
we next use these interpolated joint angles to sample the link positions for the manipulator.

.. code-block:: python

    # sample links

    # num_anchors x num_link_points x 3
    anchor_link_points = manipulator.sample_links(anchor_joint_angles, samples_per_metre=5)

    # num_anchors x num_link_points x 3
    interpolated_link_points = manipulator.sample_links(interpolated_joint_angles, samples_per_metre=5)

we show the sampled link positions during the course of the forward reaching motion in the x-y plane below.

.. image:: https://github.com/unifyai/robot/blob/master/docs/images/sampled_manipulator_links.png?raw=true
   :width: 100%

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
        <img width="75%" style="display: block;" src='https://github.com/unifyai/unifyai.github.io/blob/master/img/externally_linked/ivy_robot/demo_a.gif?raw=true'>
    </p>

**Manipulator Planning**

The second demo uses the ``MicoManipulator`` class, derived from ``Manipulator``,
to optimzie the motion of a mico robot manipulator to a set of target joint angles,
making use of functions ``ivy_robot.sample_spline_path`` and ``ivy_robot.Manipulator.sample_links``.

.. raw:: html

    <p align="center">
        <img width="75%" style="display: block;" src='https://github.com/unifyai/unifyai.github.io/blob/master/img/externally_linked/ivy_robot/demo_b.gif?raw=true'>
    </p>

Get Involved
------------

We hope the functions in this library are useful to a wide range of machine learning developers.
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
      title={Ivy: Templated deep learning for inter-framework portability},
      author={Lenton, Daniel and Pardo, Fabio and Falck, Fabian and James, Stephen and Clark, Ronald},
      journal={arXiv preprint arXiv:2102.02886},
      year={2021}
    }

# global
import os
import ivy
import time
import argparse
import ivy_mech
import ivy_robot
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from ivy.core.container import Container
from ivy_robot.manipulator import MicoManipulator
from ivy_demo_utils.ivy_scene.scene_utils import BaseSimulator
from ivy.framework_handler import set_framework, unset_framework
from ivy_demo_utils.framework_utils import choose_random_framework, get_framework_from_str


# noinspection PyProtectedMember
class Simulator(BaseSimulator):

    def __init__(self, interactive, try_use_sim):
        super().__init__(interactive, try_use_sim)

        # initialize scene
        if self.with_pyrep:
            self._spherical_vision_sensor.remove()
            for i in range(6):
                self._vision_sensors[i].remove()
                self._vision_sensor_bodies[i].remove()
                [ray.remove() for ray in self._vision_sensor_rays[i]]
            self._box.set_position(np.array([0.55, 0, 0.9]))
            self._robot.set_position(np.array([0.85003, -0.024983, 0.77837]))
            self._robot._ik_target.set_position(np.array([0, 0, -1]))
            self._robot.get_tip().set_parent(self._robot._ik_target)
            self._robot.get_tip().set_position(np.array([0, 0, -1]))
            robot_start_config = ivy.array([100., 100., 240., 180., 180., 120.])*np.pi/180
            [j.set_joint_position(p, False) for j, p in
             zip(self._robot.joints, ivy.to_numpy(robot_start_config).tolist())]
            robot_target_config = ivy.array([260., 100., 220., 0., 180., 45.])*np.pi/180
            self._robot_target.set_position(np.array([0.85003, -0.024983, 0.77837]))
            self._robot_target._ik_target.set_position(np.array([0, 0, -1]))
            self._robot_target.get_tip().set_parent(self._robot_target._ik_target)
            self._robot_target.get_tip().set_position(np.array([0, 0, -1]))
            [j.set_joint_position(p, False) for j, p in
             zip(self._robot_target.joints, ivy.to_numpy(robot_target_config).tolist())]
            self._default_camera.set_position(np.array([0.094016, -1.2767, 1.7308]))
            self._default_camera.set_orientation(np.array([i*np.pi/180 for i in [-121.32, 27.760, -164.18]]))

            input('\nScene initialized.\n\n'
                  'The simulator visualizer can be translated and rotated by clicking either the left mouse button or the wheel, '
                  'and then dragging the mouse.\n'
                  'Scrolling the mouse wheel zooms the view in and out.\n\n'
                  'You can click on any object either in the scene or the left hand panel, '
                  'then select the box icon with four arrows in the top panel of the simulator, '
                  'and then drag the object around dynamically.\n'
                  'Starting to drag and then holding ctrl allows you to also drag the object up and down.\n'
                  'Clicking the top icon with a box and two rotating arrows similarly allows rotation of the object.\n\n'
                  'The joint angles of either the robot or target robot configuration can also be changed.\n'
                  'To do this, Open the Mico or MicoTarget drop-downs on the left, and click on one of the joints "Mico_jointx", '
                  'and then click on the magnifying glass over a box on the left-most panel.\n'
                  'In the window that opens, change the value in the field Position [deg], and close the window again.\n\n'
                  'Once you have aranged the scene as desired, press enter in the terminal to continue with the demo...\n')

            # primitive scene
            self.setup_primitive_scene()

            # robot configs
            robot_start_config = ivy.array(self._robot.get_joint_positions(), 'float32')
            robot_target_config = ivy.array(self._robot_target.get_joint_positions(), 'float32')

            # ivy robot
            self._ivy_manipulator = MicoManipulator(ivy_mech.make_transformation_homogeneous(
                ivy.reshape(ivy.array(self._robot_base.get_matrix()), (3, 4))))

            # spline path
            interpolated_joint_path = ivy.transpose(ivy.linspace(robot_start_config, robot_target_config, 100), (1, 0))
            multi_spline_points = ivy.transpose(self._ivy_manipulator.sample_links(interpolated_joint_path), (1, 0, 2))
            multi_spline_sdf_vals = ivy.reshape(self.sdf(ivy.reshape(multi_spline_points, (-1, 3))), (-1, 100, 1))
            self.update_path_visualization(multi_spline_points, multi_spline_sdf_vals, None)

            # public objects
            self.ivy_manipulator = self._ivy_manipulator
            self.robot_start_config = robot_start_config
            self.robot_target_config = robot_target_config

            # wait for user input
            self._user_prompt(
                '\nInitialized scene with a robot and a target robot configuration to reach.'
                '\nPress enter in the terminal to use method ivy_robot.interpolate_spline_points '
                'to plan a spline path which reaches the target configuration whilst avoiding the objects in the scene...\n')

        else:

            # primitive scene
            self.setup_primitive_scene_no_sim(box_pos=np.array([0.55, 0, 0.9]))

            # ivy robot
            base_inv_ext_mat = ivy.array([[1, 0, 0, 0.84999895],
                                          [0, 1, 0, -0.02500308],
                                          [0, 0, 1, 0.70000124]])
            self.ivy_manipulator = MicoManipulator(ivy_mech.make_transformation_homogeneous(base_inv_ext_mat))
            self.robot_start_config = ivy.array([100., 100., 240., 180., 180., 120.])*np.pi/180
            self.robot_target_config = ivy.array([260., 100., 220., 0., 180., 45.])*np.pi/180

            # message
            print('\nInitialized dummy scene with a robot and a target robot configuration to reach.'
                  '\nClose the visualization window to use method ivy_robot.interpolate_spline_points '
                  'to plan a spline path which reaches the target configuration whilst avoiding the objects in the scene...\n')

            # plot scene before rotation
            if interactive:
                plt.imshow(mpimg.imread(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                     'msp_no_sim', 'path_0.png')))
                plt.show()

        # message
        print('\nOptimizing spline path...\n')

    def execute_motion(self, joint_configs):
        print('\nSpline path optimized.\n')
        if self._interactive:
            input('\nPress enter in the terminal to execute motion.\n')
        print('\nExecuting motion...\n')
        if self.with_pyrep:
            for i, config in enumerate(joint_configs):
                [j.set_joint_position(p, False) for j, p in zip(self._robot.joints, ivy.to_numpy(config).tolist())]
                time.sleep(0.05)
        elif self._interactive:
            this_dir = os.path.dirname(os.path.realpath(__file__))
            for i in range(11):
                plt.ion()
                plt.imshow(mpimg.imread(os.path.join(this_dir, 'msp_no_sim', 'motion_{}.png'.format(i))))
                plt.show()
                plt.pause(0.1)
                plt.ioff()


# Cost Function

def compute_length(query_vals):
    start_vals = query_vals[:, 0:-1]
    end_vals = query_vals[:, 1:]
    dists_sqrd = ivy.maximum((end_vals - start_vals)**2, 1e-12)
    distances = ivy.reduce_sum(dists_sqrd, -1)**0.5
    return ivy.reduce_mean(ivy.reduce_sum(distances, 1))


def compute_cost_and_sdfs(learnable_anchor_vals, anchor_points, start_anchor_val, end_anchor_val, query_points, sim):
    anchor_vals = ivy.concatenate((ivy.expand_dims(start_anchor_val, 0),
                                   learnable_anchor_vals, ivy.expand_dims(end_anchor_val, 0)), 0)
    joint_angles = ivy_robot.sample_spline_path(anchor_points, anchor_vals, query_points)
    link_positions = ivy.transpose(sim.ivy_manipulator.sample_links(joint_angles), (1, 0, 2))
    length_cost = compute_length(link_positions)
    sdf_vals = sim.sdf(ivy.reshape(link_positions, (-1, 3)))
    coll_cost = -ivy.reduce_mean(sdf_vals)
    total_cost = length_cost + coll_cost*10
    return total_cost[0], joint_angles, link_positions, ivy.reshape(sdf_vals, (-1, 100, 1))


def main(interactive=True, try_use_sim=True, f=None):

    # config
    this_dir = os.path.dirname(os.path.realpath(__file__))
    f = choose_random_framework(excluded=['numpy']) if f is None else f
    set_framework(f)
    sim = Simulator(interactive, try_use_sim)
    lr = 0.5
    num_anchors = 3
    num_sample_points = 100

    # spline start
    anchor_points = ivy.cast(ivy.expand_dims(ivy.linspace(0, 1, 2 + num_anchors), -1), 'float32')
    query_points = ivy.cast(ivy.expand_dims(ivy.linspace(0, 1, num_sample_points), -1), 'float32')

    # learnable parameters
    robot_start_config = ivy.array(ivy.cast(sim.robot_start_config, 'float32'))
    robot_target_config = ivy.array(ivy.cast(sim.robot_target_config, 'float32'))
    learnable_anchor_vals = ivy.variable(ivy.cast(ivy.transpose(ivy.linspace(
        robot_start_config, robot_target_config, 2 + num_anchors)[..., 1:-1], (1, 0)), 'float32'))

    # optimizer
    optimizer = ivy.SGD(lr=lr)

    # optimize
    it = 0
    colliding = True
    clearance = 0
    joint_query_vals = None
    while colliding:
        total_cost, grads, joint_query_vals, link_positions, sdf_vals = ivy.execute_with_gradients(
            lambda xs: compute_cost_and_sdfs(xs['w'], anchor_points, robot_start_config, robot_target_config,
                                             query_points, sim), Container({'w': learnable_anchor_vals}))
        colliding = ivy.reduce_min(sdf_vals[2:]) < clearance
        sim.update_path_visualization(link_positions, sdf_vals,
                                      os.path.join(this_dir, 'msp_no_sim', 'path_{}.png'.format(it)))
        learnable_anchor_vals = optimizer.step(Container({'w': learnable_anchor_vals}), grads)['w']
        it += 1
    sim.execute_motion(joint_query_vals)
    sim.close()
    unset_framework()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--non_interactive', action='store_true',
                        help='whether to run the demo in non-interactive mode.')
    parser.add_argument('--no_sim', action='store_true',
                        help='whether to run the demo without attempt to use the PyRep simulator.')
    parser.add_argument('--framework', type=str, default=None,
                        help='which framework to use. Chooses a random framework if unspecified.')
    parsed_args = parser.parse_args()
    framework = None if parsed_args.framework is None else get_framework_from_str(parsed_args.framework)
    main(not parsed_args.non_interactive, not parsed_args.no_sim, framework)

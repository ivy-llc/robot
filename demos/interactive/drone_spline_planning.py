# global
import os
import time
import argparse
import ivy_mech
import ivy_robot
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from ivy_robot.rigid_mobile import RigidMobile
from ivy_demo_utils.ivy_scene.scene_utils import BaseSimulator
from ivy_demo_utils.framework_utils import choose_random_framework, get_framework_from_str


class Simulator(BaseSimulator):

    def __init__(self, interactive, try_use_sim, f):
        super().__init__(interactive, try_use_sim, f)

        # ivy robot
        rel_body_points = f.array([[0., 0., 0.],
                                   [-0.15, -0.15, 0.],
                                   [-0.15, 0.15, 0.],
                                   [0.15, -0.15, 0.],
                                   [0.15, 0.15, 0.]])
        self.ivy_drone = RigidMobile(rel_body_points, f)

        # initialize scene
        if self.with_pyrep:
            self._spherical_vision_sensor.remove()
            for i in range(6):
                self._vision_sensors[i].remove()
                self._vision_sensor_bodies[i].remove()
                [ray.remove() for ray in self._vision_sensor_rays[i]]
            drone_start_pos = np.array([-1.15, -1.028, 0.6])
            target_pos = np.array([1.025, 1.125, 0.6])
            self._drone.set_position(drone_start_pos)
            self._drone.set_orientation(np.array([0., 0., 40])*np.pi/180)
            self._target.set_position(target_pos)
            self._target.set_orientation(np.array([0., 0., 40])*np.pi/180)
            self._default_camera.set_position(np.array([-3.2835, -0.88753, 1.3773]))
            self._default_camera.set_orientation(np.array([-151.07, 70.079, -120.45])*np.pi/180)

            input('\nScene initialized.\n\n'
                  'The simulator visualizer can be translated and rotated by clicking either the left mouse button or the wheel, '
                  'and then dragging the mouse.\n'
                  'Scrolling the mouse wheel zooms the view in and out.\n\n'
                  'You can click on any object either in the scene or the left hand panel, '
                  'then select the box icon with four arrows in the top panel of the simulator, '
                  'and then drag the object around dynamically.\n'
                  'Starting to drag and then holding ctrl allows you to also drag the object up and down.\n'
                  'Clicking the top icon with a box and two rotating arrows similarly allows rotation of the object.\n\n'
                  'Once you have aranged the scene as desired, press enter in the terminal to continue with the demo...\n')

            # primitive scene
            self.setup_primitive_scene()

            # public objects
            drone_starting_inv_ext_mat = f.array(np.reshape(self._drone.get_matrix(), (3, 4)), 'float32')
            drone_start_rot_vec_pose = ivy_mech.mat_pose_to_rot_vec_pose(drone_starting_inv_ext_mat, f=f)
            self.drone_start_pose = drone_start_rot_vec_pose
            target_inv_ext_mat = f.array(np.reshape(self._target.get_matrix(), (3, 4)), 'float32')
            target_rot_vec_pose = ivy_mech.mat_pose_to_rot_vec_pose(target_inv_ext_mat, f=f)
            self.drone_target_pose = target_rot_vec_pose

            # spline path
            drone_start_to_target_poses = f.transpose(f.linspace(
                self.drone_start_pose, self.drone_target_pose, 100), (1, 0))
            drone_start_to_target_inv_ext_mats = ivy_mech.rot_vec_pose_to_mat_pose(drone_start_to_target_poses, f=f)
            drone_start_to_target_positions =\
                f.transpose(self.ivy_drone.sample_body(drone_start_to_target_inv_ext_mats), (1, 0, 2))
            initil_sdf_vals = f.reshape(self.sdf(f.reshape(f.cast(
                drone_start_to_target_positions, 'float32'), (-1, 3))), (-1, 100, 1))
            self.update_path_visualization(drone_start_to_target_positions, initil_sdf_vals, None)

            # wait for user input
            self._user_prompt(
                '\nInitialized scene with a drone and a target position to reach.'
                '\nPress enter in the terminal to use method ivy_robot.interpolate_spline_points '
                'to plan a spline path which reaches the target whilst avoiding the objects in the scene...\n')

        else:

            # primitive scene
            self.setup_primitive_scene_no_sim()

            # public objects
            self.drone_start_pose = f.array([-1.1500, -1.0280,  0.6000,  0.0000,  0.0000,  0.6981])
            self.drone_target_pose = f.array([1.0250, 1.1250, 0.6000, 0.0000, 0.0000, 0.6981])

            # message
            print('\nInitialized dummy scene with a drone and a target position to reach.'
                  '\nClose the visualization window to use method ivy_robot.interpolate_spline_points '
                  'to plan a spline path which reaches the target whilst avoiding the objects in the scene...\n')

            # plot scene before rotation
            if interactive:
                plt.imshow(mpimg.imread(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                     'dsp_no_sim', 'path_0.png')))
                plt.show()

        print('\nOptimizing spline path...\n')

    def execute_motion(self, poses):
        print('\nSpline path optimized.\n')
        if self._interactive:
            input('\nPress enter in the terminal to execute motion.\n')
        print('\nExecuting motion...\n')
        if self.with_pyrep:
            for i in range(100):
                pose = poses[i]
                inv_ext_mat = ivy_mech.rot_vec_pose_to_mat_pose(pose)
                self._drone.set_matrix(self._f.to_numpy(inv_ext_mat).reshape((-1,)).tolist())
                time.sleep(0.05)
        elif self._interactive:
            this_dir = os.path.dirname(os.path.realpath(__file__))
            for i in range(11):
                plt.ion()
                plt.imshow(mpimg.imread(os.path.join(this_dir, 'dsp_no_sim', 'motion_{}.png'.format(i))))
                plt.show()
                plt.pause(0.1)
                plt.ioff()


# Cost Function

def compute_length(query_vals, f):
    start_vals = query_vals[0:-1]
    end_vals = query_vals[1:]
    dists_sqrd = f.maximum((end_vals - start_vals)**2, 1e-12)
    distances = f.reduce_sum(dists_sqrd, -1)**0.5
    return f.reduce_sum(distances)


def compute_cost_and_sdfs(learnable_anchor_vals, anchor_points, start_anchor_val, end_anchor_val, query_points, sim, f):
    anchor_vals = f.concatenate((f.expand_dims(start_anchor_val, 0),
                                 learnable_anchor_vals,
                                 f.expand_dims(end_anchor_val, 0)), 0)
    poses = ivy_robot.sample_spline_path(anchor_points, anchor_vals, query_points)
    inv_ext_mat_query_vals = ivy_mech.rot_vec_pose_to_mat_pose(poses, f=f)
    body_positions = f.transpose(sim.ivy_drone.sample_body(inv_ext_mat_query_vals), (1, 0, 2))
    length_cost = compute_length(body_positions, f)
    sdf_vals = sim.sdf(f.reshape(body_positions, (-1, 3)))
    coll_cost = -f.reduce_mean(sdf_vals)
    total_cost = length_cost + coll_cost * 10
    return total_cost, poses, body_positions, f.reshape(sdf_vals, (-1, 100, 1))


def main(interactive=True, try_use_sim=True, f=None):

    # config
    this_dir = os.path.dirname(os.path.realpath(__file__))
    f = choose_random_framework(excluded=['numpy']) if f is None else f
    sim = Simulator(interactive, try_use_sim, f)
    lr = 0.01
    num_anchors = 3
    num_sample_points = 100

    # 1D spline points
    anchor_points = f.cast(f.expand_dims(f.linspace(0, 1, 2 + num_anchors), -1), 'float32')
    query_points = f.cast(f.expand_dims(f.linspace(0, 1, num_sample_points), -1), 'float32')

    # learnable parameters
    drone_start_pose = f.cast(f.array(sim.drone_start_pose), 'float32')
    target_pose = f.cast(f.array(sim.drone_target_pose), 'float32')
    learnable_anchor_vals = f.variable(f.cast(f.transpose(f.linspace(
        drone_start_pose, target_pose, 2 + num_anchors)[..., 1:-1], (1, 0)), 'float32'))

    # optimize
    it = 0
    colliding = True
    clearance = 0.1
    while colliding:
        total_cost, grads, poses, body_positions, sdf_vals = f.execute_with_gradients(
            lambda xs: compute_cost_and_sdfs(xs[0], anchor_points, drone_start_pose, target_pose, query_points, sim, f),
            [learnable_anchor_vals])
        colliding = f.reduce_min(sdf_vals) < clearance
        sim.update_path_visualization(body_positions, sdf_vals,
                                      os.path.join(this_dir, 'dsp_no_sim', 'path_{}.png'.format(it)))
        learnable_anchor_vals = f.gradient_descent_update([learnable_anchor_vals], grads, lr)[0]
        it += 1
    sim.execute_motion(poses)
    sim.close()


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

"""
Collection of tests for ivy robot demos
"""

# global
import pytest


def test_demo_run_through(dev_str, fw):
    from ivy_robot_demos.run_through import main

    if fw in ["numpy", "tensorflow_graph"]:
        # numpy does not support gradients,
        # and the demo currently only supports eager mode
        pytest.skip()
    main(False, fw)


@pytest.mark.parametrize("with_sim", [False])
def test_demo_drone_spline_planning(with_sim, dev_str, fw):
    from ivy_robot_demos.interactive.drone_spline_planning import main

    if fw in ["numpy", "tensorflow_graph"]:
        # numpy does not support gradients,
        # and the demo currently only supports eager mode
        pytest.skip()
    main(False, with_sim, fw)


@pytest.mark.parametrize("with_sim", [False])
def test_demo_manipulator_spline_planning(with_sim, dev_str, fw):
    from ivy_robot_demos.interactive.manipulator_spline_planning import main

    if fw in ["numpy", "tensorflow_graph"]:
        # numpy does not support gradients,
        # and the demo currently only supports eager mode
        pytest.skip()
    main(False, with_sim, fw)

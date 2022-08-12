"""
Collection of tests for ivy robot demos
"""

# global
import pytest
from ivy_tests.test_ivy import helpers


def test_demo_run_through(dev_str, f, call, fw):
    from ivy_robot_demos.run_through import main
    if call in [helpers.np_call, helpers.tf_graph_call]:
        # numpy does not support gradients, and the demo currently only supports eager mode
        pytest.skip()
    main(False, f, fw)


@pytest.mark.parametrize(
    "with_sim", [False])
def test_demo_drone_spline_planning(with_sim, dev_str, f, call, fw):
    from ivy_robot_demos.interactive.drone_spline_planning import main
    if call in [helpers.np_call, helpers.tf_graph_call]:
        # numpy does not support gradients, and the demo currently only supports eager mode
        pytest.skip()
    main(False, with_sim, f, fw)


@pytest.mark.parametrize(
    "with_sim", [False])
def test_demo_manipulator_spline_planning(with_sim, dev_str, f, call, fw):
    from ivy_robot_demos.interactive.manipulator_spline_planning import main
    if call in [helpers.np_call, helpers.tf_graph_call]:
        # numpy does not support gradients, and the demo currently only supports eager mode
        pytest.skip()
    main(False, with_sim, f, fw)

"""
Collection of tests for ivy robot demos
"""

# global
import ivy_robot_tests.helpers as helpers


def test_demo_run_through():
    from demos.run_through import main
    for lib, call in helpers.calls:
        if call in [helpers.np_call, helpers.tf_graph_call, helpers.mx_graph_call]:
            # numpy does not support gradients, and the demo currently only supports eager mode
            continue
        main(False, f=lib)


def test_demo_drone_spline_planning():
    from demos.interactive.drone_spline_planning import main
    for lib, call in helpers.calls:
        if call in [helpers.np_call, helpers.tf_graph_call, helpers.mx_graph_call]:
            # numpy does not support gradients, and the demo currently only supports eager mode
            continue
        main(False, False, f=lib)


def test_demo_manipulator_spline_planning():
    from demos.interactive.manipulator_spline_planning import main
    for lib, call in helpers.calls:
        if call in [helpers.np_call, helpers.tf_graph_call, helpers.mx_graph_call]:
            # numpy does not support gradients, and the demo currently only supports eager mode
            continue
        main(False, False, f=lib)

#!/bin/bash -e
docker run --rm -it -v "$(pwd)":/ivy_robot unifyai/robot:latest python3 -m pytest ivy_robot_tests/

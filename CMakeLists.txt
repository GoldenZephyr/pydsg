cmake_minimum_required(VERSION 3.10)
project(pydsg)

find_package(catkin REQUIRED COMPONENTS
)

#include_directories(
#  include
#)

catkin_python_setup()

# add_service_files(
#   FILES
#   PlanToPose.srv
# )

#generate_messages(
#  DEPENDENCIES
#  std_msgs
#  geometry_msgs
#  nav_msgs
#)

#catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES active_dsg_fsm
#  CATKIN_DEPENDS
#    roscpp
#    std_msgs
#    nav_msgs
#    geometry_msgs
#    message_runtime
#    planning_ros_utils
#    jackal_mp_tracker
#    jackal_tracker_msgs
#    actionlib
#    actionlib_msgs
#)


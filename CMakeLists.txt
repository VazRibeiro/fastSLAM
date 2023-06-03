cmake_minimum_required(VERSION 3.0.2)
project(fastslam)


## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  nav_msgs
  geometry_msgs
)

#catkin_python_setup()

################################################
## Declare ROS messages, services and actions ##
################################################

## Generate messages in the 'msg' folder
add_message_files(
  DIRECTORY msg
)

## Generate services in the 'srv' folder
# add_service_files(
#   DIRECTORY srv
# )

## Generate actions in the 'action' folder
# add_action_files(
#   DIRECTORY action
# )


catkin_package(
  CATKIN_DEPENDS 
  rospy 
  std_msgs
  nav_msgs
  geometry_msgs
)

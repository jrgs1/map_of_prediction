cmake_minimum_required(VERSION 2.8.3)
project(map_building)

SET(OpenCV_DIR /home/j/prg/opencv/build)

find_package(catkin REQUIRED OpenCV REQUIRED COMPONENTS
        sensor_msgs
        cv_bridge
        roscpp
        std_msgs
        image_transport
        rosconsole
)

catkin_package()

include_directories(
        include
        ${catkin_INCLUDE_DIRS}
        ${catkin_DIR}
        ${OpenCV_INCLUDE_DIRS}
        ${catkin_DIR}/src/map_building/src
)

add_executable(${PROJECT_NAME}_node src/map_building_node.cpp)

target_link_libraries(${PROJECT_NAME}_node
        ${catkin_LIBRARIES}
        ${OpenCV_LIBS}
)

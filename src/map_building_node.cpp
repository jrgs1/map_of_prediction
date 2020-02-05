#include <ros/ros.h>
#include "MapBuilder.cpp"

int main(int argc, char** argv)
{
    ros::init(argc, argv, "image_converter");
	MapBuilder mb(0.1, 2, 5, 1, false, false);
    ros::spin();
    return 0;
}

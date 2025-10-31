#pragma once

#include "common.h"
#include "utils/file.h"
#include "utils/buffer.h"
#include "utils/imgproc.h"

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/point_field.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <cv_bridge/cv_bridge.h>
#include <builtin_interfaces/msg/time.hpp>
#include <rclcpp/time.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>

#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2/LinearMath/Quaternion.h>

#include <tf2_sensor_msgs/tf2_sensor_msgs.hpp>
#include <tf2_ros/create_timer_ros.h>

#include <tf2_ros/buffer.h>
#include <tf2_ros/message_filter.h>
#include <tf2_ros/transform_listener.h>

#include <sensor_msgs/msg/laser_scan.hpp>

using ImageMsg = sensor_msgs::msg::Image;
typedef message_filters::sync_policies::ApproximateTime<ImageMsg, ImageMsg> approximate_sync_policy;
typedef TripletBuffer<ImageMsg::SharedPtr>::Dataset BufferData;

// std::shared_ptr<TripletBuffer<ImageMsg::SharedPtr>::Dataset>

struct ImageMsgDetail
{
    std::vector<cv::Mat> images;
    uint64_t publish_time;
    uint64_t subscri_time;
};

class SubNode : public rclcpp::Node
{
public:
    SubNode(std::string node_name, std::string camera_type, nlohmann::json config) : Node(node_name), camera_type(camera_type), config(config)
    {
    }
    ~SubNode() = default;

public:
    std::string camera_type;

public:
    void InitNode(std::string topic1, std::string topic2);

    void LogInfo(const std::string &info);

    void CallBack(const sensor_msgs::msg::Image::SharedPtr &msg1, const sensor_msgs::msg::Image::SharedPtr &msg2);

    std::shared_ptr<BufferData> Read();

public:
    nlohmann::json config;
    TripletBuffer<ImageMsg::SharedPtr> buffer_;
    std::shared_ptr<message_filters::Subscriber<ImageMsg>> sub1_;
    std::shared_ptr<message_filters::Subscriber<ImageMsg>> sub2_;
    std::shared_ptr<message_filters::Synchronizer<approximate_sync_policy>> syncApproximate_;
};

class PubCloudNode : public rclcpp::Node
{
public:
    PubCloudNode(std::string node_name = "pub_node") : Node(node_name)
    {
        InitNode();
    }
    ~PubCloudNode() = default;

public:
    void InitNode(std::string pcl_topic = "stereo_point_cloud")
    {
        pointcloud2_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(pcl_topic, 10);
        // 静态坐标发布
        static_broadcaster_ = std::make_shared<tf2_ros::StaticTransformBroadcaster>(this);
        geometry_msgs::msg::TransformStamped t;
        t.header.stamp = this->now();
        t.header.frame_id = "camera_link";
        t.child_frame_id = "camera_depth_frame";

        t.transform.translation.x = 0.0;
        t.transform.translation.y = 0.0;
        t.transform.translation.z = 0.0;

        tf2::Quaternion q;
        q.setRPY(-M_PI / 2, 0, -M_PI / 2);
        q.normalize();

        t.transform.rotation.x = q.x();
        t.transform.rotation.y = q.y();
        t.transform.rotation.z = q.z();
        t.transform.rotation.w = q.w();

        static_broadcaster_->sendTransform(t);
    }

public:
    std::shared_ptr<tf2_ros::StaticTransformBroadcaster> static_broadcaster_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud2_pub_;
};

class PubLaserNode : public rclcpp::Node
{
public:
    PubLaserNode(std::string node_name = "pub_scan_node") : Node(node_name)
    {
        InitNode();
    }
    ~PubLaserNode() = default;

public:
    void InitNode(std::string scan_topic = "scan")
    {
        laserscan_pub_ = this->create_publisher<sensor_msgs::msg::LaserScan>(scan_topic, 10);

        std::string config_path = LASER;
        loadJson(config_path, config);

        tolerance_ = config["config"]["transform_tolerance"].get<double>();

        min_height_ = config["config"]["min_height"].get<double>();
        max_height_ = config["config"]["max_height"].get<double>();
        angle_min_ = config["config"]["angle_min"].get<double>();
        angle_max_ = config["config"]["angle_max"].get<double>();
        angle_increment_ = config["config"]["angle_increment"].get<double>();
        scan_time_ = config["config"]["scan_time"].get<double>();
        range_min_ = config["config"]["range_min"].get<double>();
        range_max_ = config["config"]["range_max"].get<double>();
        inf_epsilon_ = config["config"]["inf_epsilon"].get<double>();
        use_inf_ = config["config"]["use_inf"].get<bool>();
    }

public:
    double tolerance_;
    double min_height_, max_height_, angle_min_, angle_max_, angle_increment_, scan_time_, range_min_,
        range_max_;
    bool use_inf_;
    double inf_epsilon_;

    nlohmann::json config;
    rclcpp::Publisher<sensor_msgs::msg::LaserScan>::SharedPtr laserscan_pub_;
    std::unique_ptr<tf2_ros::Buffer> tf2_;
    std::unique_ptr<tf2_ros::TransformListener> tf2_listener_;
};

class NodeManage
{
public:
    NodeManage(nlohmann::json config) : config(config) {}
    ~NodeManage() = default;

public:
    void InitNode();
    void Start();
    std::shared_ptr<ImageMsgDetail> ReadImageByIndex(int index);
    void Display(bool save = false, std::string save_dir = "");

public:
    nlohmann::json config;

    std::vector<std::shared_ptr<SubNode>> sub_nodes_;

    std::shared_ptr<rclcpp::executors::MultiThreadedExecutor> executor;
    std::thread spin_thread;
};
#pragma once

#include "common.h"
#include "utils/buffer.h"
#include "utils/imgproc.h"

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <cv_bridge/cv_bridge.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

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

class PubNode : public rclcpp::Node
{
    PubNode(std::string node_name) : Node(node_name)
    {
    }
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
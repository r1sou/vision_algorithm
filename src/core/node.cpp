#include "core/node.h"

void SubNode::InitNode(std::string topic1, std::string topic2)
{
    sub1_ = std::make_shared<message_filters::Subscriber<ImageMsg>>(shared_from_this(), topic1);
    sub2_ = std::make_shared<message_filters::Subscriber<ImageMsg>>(shared_from_this(), topic2);
    syncApproximate_ = std::make_shared<message_filters::Synchronizer<approximate_sync_policy>>(approximate_sync_policy(10), *sub1_, *sub2_);
    syncApproximate_->registerCallback(&SubNode::CallBack, this);
}

void SubNode::LogInfo(const std::string &info)
{
    RCLCPP_INFO(this->get_logger(), "%s", info.c_str());
}

void SubNode::CallBack(const sensor_msgs::msg::Image::SharedPtr &msg1, const sensor_msgs::msg::Image::SharedPtr &msg2)
{
    buffer_.update(
        [&](std::vector<ImageMsg::SharedPtr> &data)
        {
            data = std::vector<ImageMsg::SharedPtr>{msg1, msg2};
        });
}

std::shared_ptr<BufferData> SubNode::Read()
{
    return buffer_.read();
}

void NodeManage::InitNode()
{
    for (auto &cfg : config["camera"])
    {
        auto sub_node = std::make_shared<SubNode>(
            cfg["camera_name"].get<std::string>(),
            cfg["camera_type"].get<std::string>(),
            cfg);
        sub_node->InitNode(
            cfg["topic"]["topic1"].get<std::string>(),
            cfg["topic"]["topic2"].get<std::string>());
        sub_nodes_.push_back(sub_node);
    }
}

void NodeManage::Start()
{
    executor = std::make_shared<rclcpp::executors::MultiThreadedExecutor>();
    for (auto &node : sub_nodes_)
    {
        executor->add_node(node);
    }
    spin_thread = std::thread(
        [this]()
        {
            executor->spin();
        });
}

std::shared_ptr<ImageMsgDetail> NodeManage::ReadImageByIndex(int index)
{

    auto data = std::make_shared<ImageMsgDetail>();

    auto buf = sub_nodes_[index]->Read();

    if (!buf || buf->data.size() == 0)
    {
        return data;
    }
    cv::Mat image1, image2;
    {
        image1 = cv::Mat(buf->data[0]->height, buf->data[0]->width, CV_8UC3, (void *)buf->data[0]->data.data(), buf->data[0]->step);
    }
    if (sub_nodes_[index]->camera_type == "dual")
    {
        image2 = cv::Mat(buf->data[1]->height, buf->data[1]->width, CV_8UC3, (void *)buf->data[1]->data.data(), buf->data[1]->step);
    }
    else
    {
        image2 = cv::Mat(buf->data[1]->height, buf->data[1]->width, CV_16UC1, (void *)buf->data[1]->data.data(), buf->data[1]->step);
    }
    if (image1.empty() || image2.empty())
    {
        return data;
    }
    if (!image1.isContinuous())
    {
        image1 = image1.clone();
    }
    if (!image2.isContinuous())
    {
        image2 = image2.clone();
    }

    auto stamp = buf->data[0]->header.stamp;
    uint64_t timestamp_ms = static_cast<uint64_t>(stamp.sec) * 1000 +
                            static_cast<uint64_t>(stamp.nanosec) / 1000000;

    data->images.push_back(image1);
    data->images.push_back(image2);
    data->publish_time = timestamp_ms;
    data->subscri_time = buf->timestamp.count();

    return data;
}

void NodeManage::Display(bool save, std::string save_dir)
{
    for (int i = 0; i < sub_nodes_.size(); i++)
    {
        auto data = ReadImageByIndex(i);
        if (data && data->images.size())
        {
            if (sub_nodes_[i]->camera_type == "dual")
            {
                cv::Mat combine;
                cv::hconcat(data->images[0], data->images[1], combine);
                cv::imshow(fmt::format("camera {}", i), combine);
            }
            else
            {
                cv::Mat combine, colormap;
                ImageRender::DepthToColorMap(data->images[1], colormap);
                cv::hconcat(data->images[0], colormap, combine);
                cv::imshow(fmt::format("camera {}", i), combine);
            }
            if (save)
            {
                time_t timestamp = time(NULL);
                std::string file_name = fmt::format("{}_{}.jpg", sub_nodes_[index]->config["camera_name"].get<std::string>(), timestamp);
                std::string file_path = save_dir + file_name;
                cv::imwrite(file_path, data->images[0]);
            }
        }
    }
    cv::waitKey(1);
}
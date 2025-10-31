#include "core/engine.h"

void Engine::InitializeCamera(nlohmann::json config)
{
    node_m = std::make_shared<NodeManage>(config);
    node_m->InitNode();
    node_m->Start();
}

bool Engine::InitializeModel(nlohmann::json config, int route)
{
    for (nlohmann::json &cfg : config["model"])
    {
        auto model = std::make_shared<BaseModel>(cfg["model_type"]);
        if (!model->Initialize(packed_dnn_handle, cfg, route))
        {
            return false;
        }
        model_m.push_back(model);
    }
    {
        detect_config = config["detect-object"];
        for (auto &[key, value] : detect_config.items())
        {
            need_detect_object_names.insert(key);
        }
    }
    return true;
}

bool Engine::InitClient(nlohmann::json config, int n_pool)
{
    publish_pool.init(n_pool);

    {
        auto websocket_config = config["websocket"];
        client_object = std::make_shared<WebSocketClient>(websocket_config);
        std::string uri = fmt::format(
            "ws://{}:{}", websocket_config["ip"].get<std::string>(), websocket_config["port"].get<std::string>());
        if (!client_object->Connect(uri))
        {
            return false;
        }
    }
    {
        auto udp_config = config["UDP"];
        std::string ip = udp_config["ip"].get<std::string>();
        uint16_t port = udp_config["port"].get<uint16_t>();
        client_laser = std::make_shared<UDPClient>(ip, port);
    }
    return true;
}

void Engine::InferenceParallel(int index, bool publish)
{
    auto data = node_m->ReadImageByIndex(index);
    if (!data->images.size())
    {
        return;
    }

    auto now = std::chrono::system_clock::now();
    int64_t start_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                             now.time_since_epoch())
                             .count();

    std::vector<ModelOutput> outputs(model_m.size());
    std::vector<std::future<void>> preprocess_task, postprocess_task;

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < model_m.size(); i++)
    {
        preprocess_task.push_back(
            preprocess_pool.enqueue(
                [&, i]()
                {
                    model_m[i]->Preprocess(data->images, index);
                }));
    }
    for (int i = 0; i < model_m.size(); i++)
    {
        preprocess_task[i].get();
        model_m[i]->Inference(index);
        postprocess_task.push_back(
            postprocess_pool.enqueue(
                [&, i]()
                {
                    model_m[i]->Postprocess(index, outputs[i]);
                }));
    }
    for (int i = 0; i < model_m.size(); i++)
    {
        postprocess_task[i].get();
    }

    std::shared_ptr<ModelOutput> output = std::make_shared<ModelOutput>();
    {
        for (auto &out : outputs)
        {
            if (out.bboxes.size())
            {
                output->bboxes.insert(
                    output->bboxes.end(),
                    std::make_move_iterator(out.bboxes.begin()),
                    std::make_move_iterator(out.bboxes.end()));
                output->names.insert(
                    output->names.end(),
                    std::make_move_iterator(out.names.begin()),
                    std::make_move_iterator(out.names.end()));
                output->scores.insert(
                    output->scores.end(),
                    std::make_move_iterator(out.scores.begin()),
                    std::make_move_iterator(out.scores.end()));
            }
            if (out.points.size())
            {
                output->points = std::move(out.points);
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    int total_delay = static_cast<int>(duration.count());
    std::cout << fmt::format(
        "detect object: {}{:2d}{}, total delay: {}{:3d}{}; ",
        ansi_colors["green"], output->bboxes.size(), ansi_colors["reset"],
        ansi_colors["red"], total_delay, ansi_colors["reset"]);

    int64_t finish_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                              end.time_since_epoch())
                              .count();
    std::cout << fmt::format(
                     "image publish: {}{}{}, start: {}{}{}, finish: {}{}{}",
                     ansi_colors["green"], data->subscri_time, ansi_colors["reset"],
                     ansi_colors["green"], start_time, ansi_colors["reset"],
                     ansi_colors["green"], finish_time, ansi_colors["reset"])
              << std::endl;
    {
        if (config["config"]["save_image"].get<bool>())
        {
            std::string save_dir = config["config"]["save_dir"].get<std::string>();
            time_t timestamp = time(NULL);
            std::string file_name = fmt::format(
                "{}_{}.jpg",
                node_m->sub_nodes_[index]->config["camera_name"].get<std::string>(),
                timestamp);
            std::string file_path = save_dir + file_name;
            cv::imwrite(file_path, data->images[0]);
        }
    }
    {
        if (publish)
        {
            if (client_object)
            {
                publish_pool.enqueue(
                    [this, output, index]()
                    {
                        PublishObject(output, index);
                    });
            }
            if (client_laser)
            {
                cv::Mat image = data->images[0].clone();
                publish_pool.enqueue(
                    [this, output, index, image]()
                    {
                        PublishPointCloud(output, image, index);
                    });
            }
        }
    }
    {
        ImageRender::DrawBox(data->images[0], output->bboxes, output->names);
        if (output->points.size())
        {
            cv::Mat colormap, combine;
            // ImageRender::DepthToColorMap(output->points, data->images[0].rows, data->images[0].cols, colormap);
            ImageRender::DisparityToColorMap(output->points, colormap, data->images[0].rows, data->images[0].cols);
            cv::vconcat(data->images[0], colormap, combine);
            cv::imshow(fmt::format("image {}", index), combine);
        }
        else
        {
            cv::imshow(fmt::format("image {}", index), data->images[0]);
        }
        cv::waitKey(1);
    }
}

void Engine::PublishObject(const std::shared_ptr<ModelOutput> &output, int index)
{
    // 目前不考虑旋转相机 TODO
    nlohmann::json camera_config = node_m->sub_nodes_[index]->config;
    nlohmann::json calib = camera_config["params"]["calibration"];
    nlohmann::json angle = camera_config["params"]["angle"];
    nlohmann::json offset = camera_config["params"]["offset"];

    int H = camera_config["shape"]["H"];
    int W = camera_config["shape"]["W"];

    nlohmann::json message;
    message["cmd_code"] = 0x12;
    message["device_id"] = camera_config["device_id"].get<int>();
    time_t timestamp = time(NULL);

    message["time_stamp"] = timestamp;
    message["key"] = JWTGenerator::generate(
        client_object->m_config["req_id"],
        client_object->m_config["key"]);
    {
        auto json = nlohmann::json::array();
        for (int i = 0; i < output->bboxes.size(); i++)
        {
            if (detect_config.contains(output->names[i]))
            {
                if (detect_config[output->names[i]]["depth"] == "center")
                {
                    int center_x = static_cast<int>(output->bboxes[i][0] + (output->bboxes[i][2] - output->bboxes[i][0]) / 2.0f);
                    int center_y = static_cast<int>(output->bboxes[i][1] + (output->bboxes[i][3] - output->bboxes[i][1]) / 2.0f);

                    if (center_y * W + center_x >= output->points.size())
                    {
                        std::cout << fmt::format(
                                         "x1y1x2y2: [{:.2f}, {:.2f}, {:.2f}, {:.2f}], points size: {}",
                                         output->bboxes[i][0],
                                         output->bboxes[i][1],
                                         output->bboxes[i][2],
                                         output->bboxes[i][3],
                                         output->points.size())
                                  << std::endl;
                        throw std::runtime_error("out of bounds!");
                    }

                    float Xw = (calib["fx"].get<float>() * calib["baseline"].get<float>()) / output->points[center_y * W + center_x];
                    if (Xw < 0.05 || Xw > 5)
                    {
                        continue;
                    }
                    Xw += offset["X"].get<float>();
                    float Yw = (calib["cx"].get<float>() - center_x) * Xw / calib["fx"].get<float>();
                    Yw += offset["Y"].get<float>();
                    float Zw = (calib["cy"].get<float>() - center_y) * Xw / calib["fy"].get<float>();
                    Zw += offset["Z"].get<float>();
                    float Ww = (output->bboxes[i][2] - output->bboxes[i][0]) * Xw / calib["fx"].get<float>();
                    float Hw = (output->bboxes[i][3] - output->bboxes[i][1]) * Xw / calib["fy"].get<float>();

                    json.push_back(
                        {{"name", output->names[i]},
                         {"obj_type", detect_config[output->names[i]]["obj_type"].get<int>()},
                         {"obj_code", detect_config[output->names[i]]["obj_code"].get<int>()},
                         {"loc", fmt::format("{:.2f},{:.2f},{:2f}", Xw, Yw, Zw)},
                         {"size", fmt::format("{:.2f},{:.2f}", Ww, Hw)}});
                }
            }
        }
        if (!json.size())
        {
            return;
        }
        message["data"] = json;
    }
    client_object->SendMsg(message.dump());
}

void Engine::PublishPointCloud(const std::shared_ptr<ModelOutput> &output, const cv::Mat &rgb, int index)
{
    if (output->points.size() == 0)
    {
        return;
    }

    nlohmann::json &camera_config = node_m->sub_nodes_[index]->config;

    int H = camera_config["shape"]["H"].get<int>();
    int W = camera_config["shape"]["W"].get<int>();

    float fx = camera_config["params"]["calibration"]["fx"].get<float>();
    float fy = camera_config["params"]["calibration"]["fy"].get<float>();
    float cx = camera_config["params"]["calibration"]["cx"].get<float>();
    float cy = camera_config["params"]["calibration"]["cy"].get<float>();
    float baseline = camera_config["params"]["calibration"]["baseline"].get<float>();

    auto pcl_msg = std::make_unique<sensor_msgs::msg::PointCloud2>(
        rosidl_runtime_cpp::MessageInitialization::SKIP);
    sensor_msgs::msg::PointCloud2 &point_cloud_msg = *pcl_msg;

    point_cloud_msg.header.stamp = pub_cloud_node->now();
    point_cloud_msg.header.frame_id = "camera_link";
    point_cloud_msg.is_dense = false;
    point_cloud_msg.fields.resize(4);
    point_cloud_msg.fields[0].name = "x";
    point_cloud_msg.fields[0].offset = 0;
    point_cloud_msg.fields[0].datatype = sensor_msgs::msg::PointField::FLOAT32;
    point_cloud_msg.fields[0].count = 1;
    point_cloud_msg.fields[1].name = "y";
    point_cloud_msg.fields[1].offset = 4;
    point_cloud_msg.fields[1].datatype = sensor_msgs::msg::PointField::FLOAT32;
    point_cloud_msg.fields[1].count = 1;
    point_cloud_msg.fields[2].name = "z";
    point_cloud_msg.fields[2].offset = 8;
    point_cloud_msg.fields[2].datatype = sensor_msgs::msg::PointField::FLOAT32;
    point_cloud_msg.fields[2].count = 1;
    point_cloud_msg.fields[3].name = "rgb";
    point_cloud_msg.fields[3].offset = 12;
    point_cloud_msg.fields[3].datatype = sensor_msgs::msg::PointField::UINT32;
    point_cloud_msg.fields[3].count = 1;

    point_cloud_msg.height = 1;
    point_cloud_msg.point_step = 16;

    int down_sample_ratio = config["config"]["down_sample_ratio"].get<int>();

    point_cloud_msg.data.resize((640 / down_sample_ratio) * (352 / down_sample_ratio) * point_cloud_msg.point_step * point_cloud_msg.height);

    float *pcd_data_ptr = reinterpret_cast<float *>(point_cloud_msg.data.data());
    uint32_t point_size = 0;
    for (int y = 0; y < H; y += down_sample_ratio)
    {
        for (int x = 0; x < W; x += down_sample_ratio)
        {
            if (output->points[y * W + x] <= 0.0f)
            {
                continue;
            }
            float depth = fx * baseline / output->points[y * W + x];
            if (depth > 5)
            {
                continue;
            }
            float X = (cx - x) / fx * depth;
            float Y = (cy - y) / fy * depth;
            if (Y < -10 || Y > 10)
            {
                continue;
            }
            *pcd_data_ptr++ = depth;
            *pcd_data_ptr++ = X;
            *pcd_data_ptr++ = Y;
            cv::Vec3b pixel = rgb.at<cv::Vec3b>(y, x);
            *(uint32_t *)pcd_data_ptr++ = (pixel[2] << 16) | (pixel[1] << 8) | (pixel[0] << 0);
            point_size++;
        }
    }
    point_cloud_msg.width = point_size;
    point_cloud_msg.row_step = point_cloud_msg.point_step * point_cloud_msg.width;
    point_cloud_msg.data.resize(point_size * point_cloud_msg.point_step *
                                point_cloud_msg.height);
    ////////////////////////////////////////////////////////////////////////////////////////////////
    auto scan_msg = std::make_unique<sensor_msgs::msg::LaserScan>();
    scan_msg->header = pcl_msg->header;

    scan_msg->angle_min = pub_laser_node->angle_min_;
    scan_msg->angle_max = pub_laser_node->angle_max_;
    scan_msg->angle_increment = pub_laser_node->angle_increment_;
    scan_msg->time_increment = 0.0;
    scan_msg->scan_time = pub_laser_node->scan_time_;
    scan_msg->range_min = pub_laser_node->range_min_;
    scan_msg->range_max = pub_laser_node->range_max_;

    uint32_t ranges_size = std::ceil(
        (scan_msg->angle_max - scan_msg->angle_min) / scan_msg->angle_increment);
    if (pub_laser_node->use_inf_)
    {
        scan_msg->ranges.assign(ranges_size, std::numeric_limits<double>::infinity());
    }
    else
    {
        scan_msg->ranges.assign(ranges_size, scan_msg->range_max + pub_laser_node->inf_epsilon_);
    }
    for (sensor_msgs::PointCloud2ConstIterator<float> iter_x(*pcl_msg, "x"),
         iter_y(*pcl_msg, "y"), iter_z(*pcl_msg, "z");
         iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z)
    {
        if (std::isnan(*iter_x) || std::isnan(*iter_y) || std::isnan(*iter_z))
        {
            continue;
        }

        if (*iter_z > pub_laser_node->max_height_ || *iter_z < pub_laser_node->min_height_)
        {
            continue;
        }

        double range = hypot(*iter_x, *iter_y);
        if (range < pub_laser_node->range_min_ || range > pub_laser_node->range_max_)
        {
            continue;
        }
        double angle = atan2(*iter_y, *iter_x);
        if (angle < scan_msg->angle_min || angle > scan_msg->angle_max)
        {
            continue;
        }

        int index_ = (angle - scan_msg->angle_min) / scan_msg->angle_increment;
        if (range < scan_msg->ranges[index_])
        {
            scan_msg->ranges[index_] = range;
        }
    }

    nlohmann::json message;

    message["cmd_code"] = 0x01;
    message["device_id"] = camera_config["device_id"].get<int>();
    time_t timestamp = time(NULL);
    message["time_stamp"] = timestamp;

    message["data"] = nlohmann::json::object();
    message["data"]["angle_min"] = scan_msg->angle_min;
    message["data"]["angle_max"] = scan_msg->angle_max;
    message["data"]["angle_increment"] = scan_msg->angle_increment;
    message["data"]["time_increment"] = scan_msg->time_increment;
    message["data"]["scan_time"] = scan_msg->scan_time;
    message["data"]["range_min"] = scan_msg->range_min;
    message["data"]["range_max"] = scan_msg->range_max;
    message["data"]["ranges"] = scan_msg->ranges;

    client_laser->SendMsg(message.dump());
    
    {
        pub_cloud_node->pointcloud2_pub_->publish(std::move(pcl_msg));
    }
    {
        pub_laser_node->laserscan_pub_->publish(std::move(scan_msg));
    }
}

void Engine::InferenceSerial(int index)
{
    auto data = node_m->ReadImageByIndex(index);
    if (!data->images.size())
    {
        return;
    }
    auto now = std::chrono::system_clock::now();
    int64_t start_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                             now.time_since_epoch())
                             .count();
    ModelOutput output;

    auto start = std::chrono::high_resolution_clock::now();
    for (auto &model : model_m)
    {
        if (node_m->config["camera"][index]["task"].contains(model->model_type))
        {
            int preprocess_time = RecordTimeCost(
                [&]()
                {
                    model->Preprocess(data->images, index);
                });
            int inference_time = RecordTimeCost(
                [&]()
                {
                    model->Inference(index);
                });
            int postprocess_time = RecordTimeCost(
                [&]()
                {
                    model->Postprocess(index, output);
                });
            std::cout << fmt::format(
                "{}{}{} model, pre: {}{:2d}{}ms, infer: {}{:2d}{}ms, post: {}{:2d}{}ms; ",
                ansi_colors["green"], model->model_type, ansi_colors["reset"],
                ansi_colors["red"], preprocess_time, ansi_colors["reset"],
                ansi_colors["red"], inference_time, ansi_colors["reset"],
                ansi_colors["red"], postprocess_time, ansi_colors["reset"]);
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    int total_delay = static_cast<int>(duration.count());
    std::cout << fmt::format(
        "detect object: {}{:2d}{}, total delay: {}{:3d}{}; ",
        ansi_colors["green"], output.bboxes.size(), ansi_colors["reset"],
        ansi_colors["red"], total_delay, ansi_colors["reset"]);

    int64_t finish_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                              end.time_since_epoch())
                              .count();

    std::cout << fmt::format(
                     "image publish: {}{}{}, start: {}{}{}, finish: {}{}{}",
                     ansi_colors["green"], data->subscri_time, ansi_colors["reset"],
                     ansi_colors["green"], start_time, ansi_colors["reset"],
                     ansi_colors["green"], finish_time, ansi_colors["reset"])
              << std::endl;

    {
        ImageRender::DrawBox(data->images[0], output.bboxes, output.names);
        if (output.points.size())
        {
            cv::Mat colormap, combine;
            ImageRender::DepthToColorMap(output.points, data->images[0].rows, data->images[0].cols, colormap);
            cv::vconcat(data->images[0], colormap, combine);
            cv::imshow("image", combine);
        }
        else
        {
            cv::imshow("image", data->images[0]);
        }
        cv::waitKey(1);
    }
}
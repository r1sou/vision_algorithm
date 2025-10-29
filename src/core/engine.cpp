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
        client_laser = std::make_shared<UDPClient>(ip,port);
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
        ImageRender::DrawBox(data->images[0], output->bboxes, output->names);
        if (output->points.size())
        {
            cv::Mat colormap, combine;
            ImageRender::DepthToColorMap(output->points, data->images[0].rows, data->images[0].cols, colormap);
            cv::hconcat(data->images[0], colormap, combine);
            cv::imshow(fmt::format("image {}", index), combine);
        }
        else
        {
            cv::imshow(fmt::format("image {}", index), data->images[0]);
        }
        cv::waitKey(1);
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
                publish_pool.enqueue(
                    [this, output, index]()
                    {
                        PublishLaserScan(output, index);
                    });
            }
        }
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

void Engine::PublishLaserScan(const std::shared_ptr<ModelOutput> &output, int index)
{
    // 目前不考虑旋转相机 TODO
    nlohmann::json &camera_config = node_m->sub_nodes_[index]->config;
    nlohmann::json message;

    message["cmd_code"] = 0x01;
    message["device_id"] = camera_config["device_id"].get<int>();
    time_t timestamp = time(NULL);
    message["time_stamp"] = timestamp;

    int H = camera_config["shape"]["H"].get<int>();
    int W = camera_config["shape"]["W"].get<int>();

    float fx = camera_config["params"]["calibration"]["fx"].get<float>();
    float fy = camera_config["params"]["calibration"]["fy"].get<float>();
    float cx = camera_config["params"]["calibration"]["cx"].get<float>();
    float cy = camera_config["params"]["calibration"]["cy"].get<float>();
    float baseline = camera_config["params"]["calibration"]["baseline"].get<float>();

    float offsetX = camera_config["params"]["offset"]["X"].get<float>();
    float offsetY = camera_config["params"]["offset"]["Y"].get<float>();
    float offsetZ = camera_config["params"]["offset"]["Z"].get<float>();

    LaserScan scan;
    for (int row = 0; row < H; row++)
    {
        for (int col = 0; col < W; col++)
        {
            if (output->points[row * W + col] <= 0.0f || output->points[row * W + col] > W)
            {
                continue;
            }

            float Xw = fx * baseline / output->points[row * W + col];
            float Yw = (cx - col) * Xw / fx;
            float Zw = (cy - row) * Xw / fy;
            if ((Zw + offsetZ) > scan.max_height || (Zw + offsetZ) < scan.min_height)
            {
                continue;
            }
            float range = hypot(Xw, Yw);
            if (range > scan.range_max || range < scan.range_min)
            {
                continue;
            }
            float angle = atan2(Yw, Xw);
            if (angle > scan.angle_max || angle < scan.angle_min)
            {
                continue;
            }

            int index = (angle - scan.angle_min) / scan.angle_increment;
            if (range < scan.ranges[index])
            {
                scan.ranges[index] = range;
            }
        }
    }
    message["data"] = nlohmann::json::object();
    message["data"]["angle_min"] = scan.angle_min;
    message["data"]["angle_max"] = scan.angle_max;
    message["data"]["angle_increment"] = scan.angle_increment;
    message["data"]["time_increment"] = scan.time_increment;
    message["data"]["scan_time"] = scan.scan_time;
    message["data"]["range_min"] = scan.range_min;
    message["data"]["range_max"] = scan.range_max;
    message["data"]["ranges"] = scan.ranges;

    client_laser->SendMsg(message.dump());
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
            cv::hconcat(data->images[0], colormap, combine);
            cv::imshow("image", combine);
        }
        else
        {
            cv::imshow("image", data->images[0]);
        }
        cv::waitKey(1);
    }
}
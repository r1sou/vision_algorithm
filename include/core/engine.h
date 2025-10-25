#pragma once

#include "model.h"
#include "node.h"

class Engine
{
public:
    Engine()
    {
        preprocess_pool.init(2);
        postprocess_pool.init(2);
    }
    ~Engine()
    {
        hbDNNRelease(packed_dnn_handle);
    }

public:
    void InitializeCamera(nlohmann::json config)
    {
        node_m = std::make_shared<NodeManage>(config);
        node_m->InitNode();
        node_m->Start();
    }

    bool InitializeModel(nlohmann::json config, int route = 4)
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

    void InferenceParallel(int index = 0)
    {
        auto data = node_m->ReadImageByIndex(index);
        if (!data->images.size())
        {
            return;
        }
        std::vector<ModelOutput> outputs(model_m.size());
        std::vector<std::future<void>> preprocess_task, postprocess_task;

        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < model_m.size(); i++)
        {
            preprocess_task.push_back(
                preprocess_pool.enqueue(
                    [&,i]()
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
                    [&,i]()
                    {
                        model_m[i]->Postprocess(index, outputs[i]);
                    }));
        }
        for (int i = 0; i < model_m.size(); i++)
        {
            postprocess_task[i].get();
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        int total_delay = static_cast<int>(duration.count());
        std::cout << fmt::format(
            "total delay: {}{:3d}{}; ",
            ansi_colors["red"], total_delay, ansi_colors["reset"]);

        auto now = std::chrono::system_clock::now();
        int64_t finish_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                                  now.time_since_epoch())
                                  .count();

        std::cout << fmt::format(
                         "image publish: {}{}{},finish: {}{}{}",
                         ansi_colors["green"], data->subscri_time, ansi_colors["reset"],
                         ansi_colors["green"], finish_time, ansi_colors["reset"])
                  << std::endl;

        ModelOutput output;
        {
            for (auto &out : outputs)
            {
                if (out.bboxes.size())
                {
                    output.bboxes.insert(
                        output.bboxes.end(),
                        std::make_move_iterator(out.bboxes.begin()),
                        std::make_move_iterator(out.bboxes.end())
                    );
                    output.names.insert(
                        output.names.end(),
                        std::make_move_iterator(out.names.begin()),
                        std::make_move_iterator(out.names.end())
                    );
                    output.scores.insert(
                        output.scores.end(),
                        std::make_move_iterator(out.scores.begin()),
                        std::make_move_iterator(out.scores.end())
                    );
                }
                if (out.points.size())
                {
                    output.points = std::move(out.points);
                }
            }
        }

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

    void InferenceSerial(int index = 0)
    {
        auto data = node_m->ReadImageByIndex(index);
        if (!data->images.size())
        {
            return;
        }
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
            "total delay: {}{:3d}{}; ",
            ansi_colors["red"], total_delay, ansi_colors["reset"]);

        auto now = std::chrono::system_clock::now();
        int64_t finish_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                                  now.time_since_epoch())
                                  .count();

        std::cout << fmt::format(
                         "image publish: {}{}{},finish: {}{}{}",
                         ansi_colors["green"], data->subscri_time, ansi_colors["reset"],
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

public:
    nlohmann::json detect_config;
    std::unordered_set<std::string> need_detect_object_names;

private:
    hbPackedDNNHandle_t packed_dnn_handle;

    ThreadPool preprocess_pool, postprocess_pool;

    std::shared_ptr<NodeManage> node_m;
    std::vector<std::shared_ptr<BaseModel>> model_m;
};

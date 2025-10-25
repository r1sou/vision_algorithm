#include "core/engine.h"

void testModel(bool parallel)
{
    nlohmann::json camera_config, model_config;
    std::string camera_config_file = CAMERA;
    std::string camera_model_file = MODEL;

    loadJson(camera_config_file, camera_config);
    loadJson(camera_model_file, model_config);

    auto engine = std::make_shared<Engine>();
    engine->InitializeCamera(camera_config);
    engine->InitializeModel(model_config);

    while (rclcpp::ok())
    {
        if (parallel)
        {
            engine->InferenceParallel();
        }
        else
        {
            engine->InferenceSerial();
        }
    }
}

int main(int argc, char *argv[])
{

    rclcpp::init(argc, argv);

    bool parallel = argc > 1 ? true : false;

    testModel(parallel);

    cv::destroyAllWindows();
    rclcpp::shutdown();

    return 0;
}

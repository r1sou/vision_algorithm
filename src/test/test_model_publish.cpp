#include "core/engine.h"

void testModel()
{
    nlohmann::json camera_config, model_config,client_config;
    std::string camera_config_file = CAMERA;
    std::string model_config_file = MODEL;
    std::string client_config_file = CLIENT;

    loadJson(camera_config_file, camera_config);
    loadJson(model_config_file, model_config);
    loadJson(client_config_file, client_config);

    auto engine = std::make_shared<Engine>();
    engine->InitializeCamera(camera_config);
    engine->InitializeModel(model_config);
    engine->InitClient(client_config);

    while (rclcpp::ok())
    {
        try{
            engine->Inference();
        }
        catch(...){
            break;
        }
    }
}

int main(int argc, char *argv[])
{

    rclcpp::init(argc, argv);

    testModel();

    cv::destroyAllWindows();
    rclcpp::shutdown();

    return 0;
}

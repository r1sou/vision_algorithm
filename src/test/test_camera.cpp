#include "core/node.h"
#include "utils/file.h"

void testNode()
{
    nlohmann::json camera_config,config;
    std::string camera_config_file = CAMERA;
    std::string config_file = CONFIG;

    loadJson(camera_config_file, camera_config);
    loadJson(config_file,config);

    auto node = std::make_shared<NodeManage>(camera_config);
    node->InitNode();
    node->Start();

    bool save = config["config"]["save_image"].get<bool>();
    std::string save_dir = config["config"]["save_dir"].get<std::string>();
    if(save){
        createDir(save_dir);
    }


    while (rclcpp::ok())
    {
        node->Display(save,save_dir);
    }
}

int main(int argc, char *argv[])
{

    rclcpp::init(argc, argv);

    testNode();

    cv::destroyAllWindows();
    rclcpp::shutdown();

    return 0;
}

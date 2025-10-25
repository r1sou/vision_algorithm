#include "core/node.h"
#include "utils/file.h"

void testNode()
{
    nlohmann::json camera_config;
    std::string camera_config_file = CAMERA;

    loadJson(camera_config_file, camera_config);

    auto node = std::make_shared<NodeManage>(camera_config);
    node->InitNode();
    node->Start();

    while (rclcpp::ok())
    {
        node->Display();
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

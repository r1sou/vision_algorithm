#include "core/node.h"
#include "utils/file.h"

int main(int argc, char *argv[])
{

    rclcpp::init(argc, argv);

    std::cout<<"main !!!"<<std::endl;

    cv::destroyAllWindows();
    rclcpp::shutdown();

    return 0;
}

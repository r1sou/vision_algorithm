#include "common.h"
#include "utils/file.h"

struct ModelOutput
{
    std::vector<float> scores;
    std::vector<std::string> names;
    std::vector<std::vector<float>> bboxes;

    std::vector<float> points;
};

struct ModelParam
{
    int class_nums = 80;
    std::vector<std::string> class_names = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
        "cell phone", "microwave", "oven", "toaster", "sink",
        "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"};
    int input_H = 640, input_W = 640;
    float conf_thresh = 0.5, iou_thresh = 0.7;
    int REG = 16, nms_top_k = 300;
};

struct LaserScan
{
    double angle_min = -1.5708;       // start angle of the scan [rad]
    double angle_max = 1.5708;        // end angle of the scan [rad]
    double angle_increment = 0.00315; // angular distance between measurements [rad]
    double time_increment = 0.1;      // time between measurements [seconds] - if your scanner is moving, this will be used in interpolating position of 3d points
    double scan_time = 0.1;           // time between scans [seconds]
    double min_height = 0.0;
    double max_height = 1.0;
    double range_min = 0.05;          // minimum range value [m]
    double range_max = 150.0;         // maximum range value [m]
    std::vector<double> ranges;       // range data [m] (Note: values < range_min or > range_max should be discarded)
    std::vector<double> intensities;  // intensity data [device-specific units].  If your device does not provide intensities, please leave the array empty.

    LaserScan()
    {
        std::string file = LASERSCAN;
        nlohmann::json params;
        loadJson(file,params);

        angle_min = params["laserscan"].value("angle_min",angle_min);
        angle_max = params["laserscan"].value("angle_max",angle_max);
        angle_increment = params["laserscan"].value("angle_increment",angle_increment);
        time_increment = params["laserscan"].value("time_increment",time_increment);
        scan_time = params["laserscan"].value("scan_time",scan_time);
        min_height = params["laserscan"].value("min_height",min_height);
        max_height = params["laserscan"].value("max_height",max_height);
        range_min = params["laserscan"].value("range_min",range_min);
        range_max = params["laserscan"].value("range_max",range_max);

        uint32_t ranges_size = std::ceil((angle_max - angle_min) / angle_increment);
        ranges.assign(ranges_size, range_max + 1e-4f);
    }
};

#pragma once

#include "common.h"
#include "utils/file.h"
#include "utils/imgproc.h"

#include "dnn/hb_dnn.h"
#include "dnn/hb_sys.h"

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

struct ModelOutput
{
    std::vector<float> scores;
    std::vector<std::string> names;
    std::vector<std::vector<float>> bboxes;

    std::vector<float> points;
};

class BaseModel
{
public:
    BaseModel(std::string model_type) : model_type(model_type) {}
    ~BaseModel()
    {
        release();
    }

public:
    
    bool Initialize(hbPackedDNNHandle_t &packed_dnn_handle, nlohmann::json config, int route);

    bool InitModel(hbPackedDNNHandle_t &packed_dnn_handle, std::string model_path);

    bool InitTensor(int route);

    void InitParam(nlohmann::json config);

    bool AllocateInputTensor();

    bool AllocateOutputTensor();

    void CalcuAllocateMemorySize();

    void release();

    void Preprocess(std::vector<cv::Mat> &images, int tensor_index);

    void Inference(int tensor_index);

    void Postprocess(int tensor_index, ModelOutput &output);

    void YOLOPostprocess(int tensor_index, ModelOutput &output);

    void DecodeBox(
        float *cls_ptr, int32_t *box_ptr, float *scale_ptr, int H, int W, float stride,
        std::vector<std::vector<cv::Rect2d>> &bboxes, std::vector<std::vector<float>> &scores);

    void StereoPostprocess(int tensor_index, ModelOutput &output);

public:
    ModelParam param;
    std::string model_type;
    std::vector<std::vector<float>> scalers;
    float input_memory = 0.0, output_memory = 0.0;

public:
    hbDNNHandle_t dnn_handle;
    std::vector<std::vector<hbDNNTensor>> batch_input_tensor;
    std::vector<std::vector<hbDNNTensor>> batch_output_tensor;
};
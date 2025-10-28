#pragma once

#include "common.h"
#include "utils/file.h"
#include "utils/type.h"
#include "utils/imgproc.h"

#include "dnn/hb_dnn.h"
#include "dnn/hb_sys.h"

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
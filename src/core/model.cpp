#include "core/model.h"

bool BaseModel::Initialize(hbPackedDNNHandle_t &packed_dnn_handle, nlohmann::json config, int route)
{
    std::string model_path = config["model_path"].get<std::string>();

    if (!isFileExist(model_path))
    {
        return false;
    }
    {
        if (!InitModel(packed_dnn_handle, model_path))
        {
            return false;
        }
        InitParam(config);
    }
    {
        if (!InitTensor(route))
        {
            return false;
        }
        scalers.resize(route);
    }

    CalcuAllocateMemorySize();
    std::string info = fmt::format(
        "[Model] load model from {}{}{} successful, allocate input tensor total memory: {}{:.2f}{}MB, allocate output tensor total memory: {}{:.2f}{}MB",
        ansi_colors["green"], model_path, ansi_colors["reset"],
        ansi_colors["red"], input_memory, ansi_colors["reset"],
        ansi_colors["red"], output_memory, ansi_colors["reset"]);
    std::cout << info << std::endl;
    return true;
}

bool BaseModel::InitModel(hbPackedDNNHandle_t &packed_dnn_handle, std::string model_path)
{
    try
    {
        const char *model_path_c = model_path.c_str();
        int model_count = 0;
        const char **model_name_list;
        hbDNNInitializeFromFiles(&packed_dnn_handle, &model_path_c, 1);
        hbDNNGetModelNameList(&model_name_list, &model_count, packed_dnn_handle);
        hbDNNGetModelHandle(&dnn_handle, packed_dnn_handle, model_name_list[0]);
    }
    catch (...)
    {
        std::cerr << "InitModel failed" << std::endl;
        return false;
    }
    return true;
}

bool BaseModel::InitTensor(int route)
{
    batch_input_tensor.resize(route);
    batch_output_tensor.resize(route);
    return AllocateInputTensor() && AllocateOutputTensor();
}

void BaseModel::InitParam(nlohmann::json config)
{
    param.input_H = config["input_size"][0];
    param.input_W = config["input_size"][1];

    param.class_names = config.value("class_names", param.class_names);
    param.class_nums = param.class_names.size();
    param.conf_thresh = config.value("conf_thresh", param.conf_thresh);
    param.iou_thresh = config.value("iou_thresh", param.iou_thresh);
    param.REG = config.value("REG", param.REG);
    param.nms_top_k = config.value("nms_top_k", param.nms_top_k);
}

bool BaseModel::AllocateInputTensor()
{
    try
    {
        int input_count;
        hbDNNGetInputCount(&input_count, dnn_handle);

        for (auto &input_tensor : batch_input_tensor)
        {
            for (int i = 0; i < input_count; i++)
            {
                hbDNNTensor tensor;
                hbDNNGetInputTensorProperties(&tensor.properties, dnn_handle, i);
                if (tensor.properties.tensorType == HB_DNN_IMG_TYPE_NV12)
                {
                    int32_t batch = tensor.properties.alignedShape.dimensionSize[0];
                    int32_t batch_size = tensor.properties.alignedByteSize / batch;

                    tensor.properties.alignedByteSize = batch_size;
                    tensor.properties.validShape.dimensionSize[0] = 1;
                    tensor.properties.alignedShape = tensor.properties.validShape;

                    for (int j = 0; j < batch; j++)
                    {
                        hbSysAllocCachedMem(&tensor.sysMem[0], batch_size);
                        input_tensor.push_back(tensor);
                    }
                }
                else if (tensor.properties.tensorType == HB_DNN_IMG_TYPE_NV12_SEPARATE)
                {
                    int32_t batch = tensor.properties.alignedShape.dimensionSize[0];
                    int32_t batch_size = tensor.properties.alignedByteSize / batch;

                    tensor.properties.alignedByteSize = batch_size;
                    tensor.properties.validShape.dimensionSize[0] = 1;
                    tensor.properties.alignedShape = tensor.properties.validShape;

                    for (int j = 0; j < batch; j++)
                    {
                        hbSysAllocCachedMem(&tensor.sysMem[0], batch_size * 2 / 3);
                        hbSysAllocCachedMem(&tensor.sysMem[1], batch_size * 1 / 3);
                        input_tensor.push_back(tensor);
                    }
                }
                else
                {
                    int input_memSize = tensor.properties.alignedByteSize;
                    hbSysAllocCachedMem(&tensor.sysMem[0], input_memSize);
                    tensor.properties.alignedShape = tensor.properties.validShape;
                    input_tensor.push_back(tensor);
                }
            }
        }
    }
    catch (...)
    {
        std::cerr << "allocate input tensor failed" << std::endl;
        return false;
    }
    return true;
}

bool BaseModel::AllocateOutputTensor()
{
    try
    {
        int output_count;
        hbDNNGetOutputCount(&output_count, dnn_handle);
        for (auto &output_tensor : batch_output_tensor)
        {
            for (int i = 0; i < output_count; i++)
            {
                hbDNNTensor tensor;
                hbDNNGetOutputTensorProperties(&tensor.properties, dnn_handle, i);
                int output_memSize = tensor.properties.alignedByteSize;
                hbSysAllocCachedMem(&tensor.sysMem[0], output_memSize);
                output_tensor.push_back(tensor);
            }
        }
    }
    catch (...)
    {
        std::cerr << "allocate output tensor failed" << std::endl;
        return false;
    }
    return true;
}

void BaseModel::CalcuAllocateMemorySize()
{
    input_memory = 0.0, output_memory = 0.0;

    for (auto &tensor : batch_input_tensor[0])
    {
        input_memory += (1.0 * tensor.properties.alignedByteSize) / 1024 / 1024;
    }
    output_memory *= batch_input_tensor.size();

    for (auto &tensor : batch_output_tensor[0])
    {
        output_memory += (1.0 * tensor.properties.alignedByteSize) / 1024 / 1024;
    }
    output_memory *= batch_output_tensor.size();
}

void BaseModel::release()
{
    std::cout << "release all tensor" << std::endl;
    for (auto &output_tensor : batch_input_tensor)
    {
        for (auto &tensor : output_tensor)
        {
            hbSysFreeMem(&tensor.sysMem[0]);
        }
    }
    for (auto &output_tensor : batch_output_tensor)
    {
        for (auto &tensor : output_tensor)
        {
            hbSysFreeMem(&tensor.sysMem[0]);
        }
    }
}

void BaseModel::Preprocess(std::vector<cv::Mat> &images, int tensor_index)
{
    auto &scale = scalers[tensor_index];

    bool isYOLO = (model_type == "yolo") ? true : false;

    int n = 2 - isYOLO;

    for (int i = 0; i < n; i++)
    {
        cv::Mat resize_image, input_image;
        if (isYOLO)
        {
            ImageProc::LetterBox(images[i], resize_image, param.input_W, param.input_H, scale);
        }
        else
        {
            ImageProc::Resize(images[i], resize_image, param.input_W, param.input_H, scale);
        }
        ImageProc::BGR2NV12(resize_image, input_image);

        auto &tensor = batch_input_tensor[tensor_index][i];
        hbSysWriteMem(&tensor.sysMem[0], (char *)input_image.data, input_image.rows * input_image.cols);
        hbSysFlushMem(&tensor.sysMem[0], HB_SYS_MEM_CACHE_CLEAN);
    }
}

void BaseModel::Inference(int tensor_index)
{
    auto &input_tensor = batch_input_tensor[tensor_index];
    auto &output_tensor = batch_output_tensor[tensor_index];

    hbDNNTaskHandle_t task_handle = nullptr;
    hbDNNInferCtrlParam infer_ctrl_param;
    HB_DNN_INITIALIZE_INFER_CTRL_PARAM(&infer_ctrl_param);

    hbDNNTensor *in_ptr = &input_tensor[0];
    hbDNNTensor *out_ptr = &output_tensor[0];
    hbDNNInfer(&task_handle, &out_ptr, in_ptr, dnn_handle, &infer_ctrl_param);

    hbDNNWaitTaskDone(task_handle, 0);
    hbDNNReleaseTask(task_handle);
}

void BaseModel::Postprocess(int tensor_index, ModelOutput &output)
{
    if (model_type == "yolo")
    {
        YOLOPostprocess(tensor_index, output);
    }
    else
    {
        StereoPostprocess(tensor_index, output);
    }
}

void BaseModel::StereoPostprocess(int tensor_index, ModelOutput &output)
{
    auto &output_tensor = batch_output_tensor[tensor_index];

    for (auto &tensor : output_tensor)
    {
        hbSysFlushMem(&tensor.sysMem[0], HB_SYS_MEM_CACHE_INVALIDATE);
    }

    int32_t *disp_shape = output_tensor[0].properties.validShape.dimensionSize;
    int disp_c_dim = disp_shape[1];
    int disp_h_dim = disp_shape[2];
    int disp_w_dim = disp_shape[3];
    int total_disp_size = disp_h_dim * disp_w_dim;

    int32_t *spx_shape = output_tensor[1].properties.validShape.dimensionSize;
    int spx_c_dim = spx_shape[1];
    int spx_h_dim = spx_shape[2];
    int spx_w_dim = spx_shape[3];
    int total_size = spx_h_dim * spx_w_dim;
    int32_t scale_h = spx_h_dim / disp_h_dim, scale_w = spx_w_dim / disp_w_dim;

    float scale_constant = 1.0;
    float scale_factor;
    float *disp_scale = &scale_constant;
    float *spx_scale = &scale_constant;
    if (output_tensor[0].properties.quantiType == SCALE)
    {
        disp_scale = output_tensor[0].properties.scale.scaleData;
    }
    if (output_tensor[1].properties.quantiType == SCALE)
    {
        spx_scale = output_tensor[1].properties.scale.scaleData;
    }
    scale_factor = (*disp_scale * *spx_scale);

    output.points.resize(total_size, 0.f);
    float *result_ptr = output.points.data();
    if (output_tensor[0].properties.tensorType == HB_DNN_TENSOR_TYPE_S32 && output_tensor[1].properties.tensorType == HB_DNN_TENSOR_TYPE_S16)
    {
        int32_t *disp = reinterpret_cast<int32_t *>(output_tensor[0].sysMem[0].virAddr);
        int16_t *spx = reinterpret_cast<int16_t *>(output_tensor[1].sysMem[0].virAddr);

        for (int32_t i = 0; i < spx_c_dim; ++i)
        {
            for (int32_t y = 0; y < spx_h_dim; ++y)
            {
                int32_t idx_y = y / scale_h;
                int32_t output_offset = spx_w_dim * y;
                for (int32_t x = 0; x < spx_w_dim; x += 4)
                {
                    int32_t idx_x = x / scale_w;

                    int16x4_t spx_s16 = vld1_s16(&spx[y * spx_w_dim + x]);
                    int32x4_t spx_s32 = vmovl_s16(spx_s16);

                    int32_t disp_val_scalar = disp[idx_y * disp_w_dim + idx_x];
                    int32x4_t disp_s32 = vdupq_n_s32(disp_val_scalar);

                    float32x4_t spx_f32 = vcvtq_f32_s32(spx_s32);
                    float32x4_t disp_f32 = vcvtq_f32_s32(disp_s32);

                    float32x4_t mul_result = vmulq_f32(disp_f32, spx_f32);
                    float32x4_t current_output = vld1q_f32(&result_ptr[output_offset + x]);
                    float32x4_t updated_output = vaddq_f32(current_output, mul_result);
                    vst1q_f32(&result_ptr[output_offset + x], updated_output);
                }
            }
            disp += total_disp_size;
            spx += total_size;
        }

        if (scale_factor != 1.0f)
        {
            for (int32_t j = 0; j < total_size; j += 4)
            {
                vst1q_f32(result_ptr + j, vmulq_n_f32(vld1q_f32(result_ptr + j), scale_factor));
            }
        }
    }
    else if (output_tensor[0].properties.tensorType == HB_DNN_TENSOR_TYPE_F32 && output_tensor[1].properties.tensorType == HB_DNN_TENSOR_TYPE_F32)
    {
        float *disp = reinterpret_cast<float *>(output_tensor[0].sysMem[0].virAddr);
        float *spx = reinterpret_cast<float *>(output_tensor[1].sysMem[0].virAddr);

        for (int32_t i = 0; i < spx_c_dim; ++i)
        {
            for (int32_t y = 0; y < spx_h_dim; ++y)
            {
                int32_t idx_y = y / scale_h;
                int32_t output_offset = spx_w_dim * y;
                for (int32_t x = 0; x < spx_w_dim; x += 4)
                {
                    int32_t idx_x = x / scale_w;

                    float32x4_t spx_f32 = vld1q_f32(&spx[y * spx_w_dim + x]);

                    float disp_val_scalar = disp[idx_y * disp_w_dim + idx_x];
                    float32x4_t disp_f32 = vdupq_n_f32(disp_val_scalar);

                    float32x4_t mul_result = vmulq_f32(disp_f32, spx_f32);
                    float32x4_t current_output = vld1q_f32(&result_ptr[output_offset + x]);
                    float32x4_t updated_output = vaddq_f32(current_output, mul_result);
                    vst1q_f32(&result_ptr[output_offset + x], updated_output);
                }
            }

            disp += total_disp_size;
            spx += total_size;
        }
        if (scale_factor != 1.0f)
        {
            for (int32_t j = 0; j < total_size; j += 4)
            {
                float32x4_t cur = vld1q_f32(result_ptr + j);
                float32x4_t scaled = vmulq_n_f32(cur, scale_factor);
                vst1q_f32(result_ptr + j, scaled);
            }
        }
    }
}

void BaseModel::YOLOPostprocess(int tensor_index, ModelOutput &output)
{
    auto scale = scalers[tensor_index];

    std::vector<std::vector<cv::Rect2d>> bboxes;
    std::vector<std::vector<float>> scores;
    bboxes.resize(param.class_nums);
    scores.resize(param.class_nums);

    auto &output_tensor = batch_output_tensor[tensor_index];

    for (auto &tensor : output_tensor)
    {
        hbSysFlushMem(&tensor.sysMem[0], HB_SYS_MEM_CACHE_INVALIDATE);
    }

    DecodeBox(
        reinterpret_cast<float *>(output_tensor[0].sysMem[0].virAddr),
        reinterpret_cast<int32_t *>(output_tensor[1].sysMem[0].virAddr),
        reinterpret_cast<float *>(output_tensor[1].properties.scale.scaleData),
        80, 80, 8.0,
        bboxes, scores);
    DecodeBox(
        reinterpret_cast<float *>(output_tensor[2].sysMem[0].virAddr),
        reinterpret_cast<int32_t *>(output_tensor[3].sysMem[0].virAddr),
        reinterpret_cast<float *>(output_tensor[3].properties.scale.scaleData),
        40, 40, 16.0,
        bboxes, scores);
    DecodeBox(
        reinterpret_cast<float *>(output_tensor[4].sysMem[0].virAddr),
        reinterpret_cast<int32_t *>(output_tensor[5].sysMem[0].virAddr),
        reinterpret_cast<float *>(output_tensor[5].properties.scale.scaleData),
        20, 20, 32.0,
        bboxes, scores);

    std::vector<std::vector<int>> indices(param.class_nums);
    for (int i = 0; i < param.class_nums; i++)
    {
        cv::dnn::NMSBoxes(bboxes[i], scores[i], param.conf_thresh, param.iou_thresh, indices[i], 1.f, param.nms_top_k);
    }
    for (int cls_id = 0; cls_id < param.class_nums; cls_id++)
    {
        for (auto index : indices[cls_id])
        {
            std::string name = param.class_names[cls_id];

            float x1 = (bboxes[cls_id][index].x - scale[2]) / scale[0];
            float y1 = (bboxes[cls_id][index].y - scale[3]) / scale[1];
            float x2 = x1 + (bboxes[cls_id][index].width) / scale[0];
            float y2 = y1 + (bboxes[cls_id][index].height) / scale[1];

            output.bboxes.push_back(std::vector<float>{x1, y1, x2, y2});
            output.scores.push_back(scores[cls_id][index]);
            output.names.push_back(name);
        }
    }
}

void BaseModel::DecodeBox(
    float *cls_ptr, int32_t *box_ptr, float *scale_ptr,
    int H, int W, float stride,
    std::vector<std::vector<cv::Rect2d>> &bboxes,
    std::vector<std::vector<float>> &scores)
{
    float conf_thres = -log(1 / param.conf_thresh - 1);
    for (int h = 0; h < H; ++h)
    {
        for (int w = 0; w < W; ++w)
        {
            int cls_id = 0;
            for (int i = 1; i < param.class_nums; ++i)
                if (cls_ptr[i] > cls_ptr[cls_id])
                    cls_id = i;
            if (cls_ptr[cls_id] < conf_thres)
            {
                cls_ptr += param.class_nums;
                box_ptr += param.REG * 4;
                continue;
            }
            float score = 1.f / (1.f + std::exp(-cls_ptr[cls_id]));
            float ltrb[4] = {0.f};
            for (int k = 0; k < 4; ++k)
            {
                float sum = 0.f;
                for (int j = 0; j < param.REG; ++j)
                {
                    float d = std::exp(float(box_ptr[k * param.REG + j]) * scale_ptr[k * param.REG + j]);
                    ltrb[k] += d * j;
                    sum += d;
                }
                ltrb[k] /= sum;
            }
            if (ltrb[2] + ltrb[0] <= 0 || ltrb[3] + ltrb[1] <= 0)
            {
                cls_ptr += param.class_nums;
                box_ptr += param.REG * 4;
                continue;
            }
            float x1 = (w + 0.5f - ltrb[0]) * stride;
            float y1 = (h + 0.5f - ltrb[1]) * stride;
            float x2 = (w + 0.5f + ltrb[2]) * stride;
            float y2 = (h + 0.5f + ltrb[3]) * stride;

            bboxes[cls_id].emplace_back(x1, y1, x2 - x1, y2 - y1);
            scores[cls_id].push_back(score);
            cls_ptr += param.class_nums;
            box_ptr += param.REG * 4;
        }
    }
}

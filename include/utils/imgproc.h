#pragma once

#include "common.h"

#ifdef __aarch64__
#include <arm_neon.h>
#endif

struct ImageProc
{
    static void Resize(cv::Mat &src, cv::Mat &dst, int width, int height);
    static void Resize(cv::Mat &src, cv::Mat &dst, int width, int height, std::vector<float> &scale);

    static void LetterBox(cv::Mat &src, cv::Mat &dst, int width, int height);
    static void LetterBox(cv::Mat &src, cv::Mat &dst, int width, int height, std::vector<float> &scale);

    static void BGR2RGB(cv::Mat &src, cv::Mat &dst);

    static void MinMaxNormalize(cv::Mat &src);
    static void MinMaxNormalize(cv::Mat &src, cv::Mat &dst);

#ifdef __aarch64__
    static void BGR2NV12_neon(uint8_t *src, uint8_t *dst, int width, int height);
    static void BGR2NV12(cv::Mat &src, cv::Mat &dst);
#endif
};

struct ImageRender
{
    static void DrawBox(cv::Mat &img, std::vector<float> &box, std::string &label);
    static void DrawBox(cv::Mat &img, std::vector<std::vector<float>> &boxes, std::vector<std::string> &labels);
    static void DepthToColorMap(std::vector<float> &depth, int H, int W, cv::Mat &color_map, bool smooth_for_visualization = true);
    static void DepthToColorMap(cv::Mat &depth, cv::Mat &color_map);

    static void DisparityToColorMap(std::vector<float> &points, cv::Mat &colormap, int H, int W)
    {
        cv::Mat disp_mat(H, W, CV_32FC1, const_cast<float *>(points.data()));
        disp_mat.convertTo(colormap, CV_8UC1, 3, 0);
        cv::applyColorMap(colormap,colormap,cv::COLORMAP_JET);
    }

    static void DisparityToDepth(std::vector<float> &points, cv::Mat &depth, int H, int W, float camera_fx, float baseline)
    {
        depth = cv::Mat(H, W, CV_16UC1);
        uint16_t *depth_data = (uint16_t *)depth.data;
        float factor = 1000 * (camera_fx * baseline);
        uint32_t num_pixels = points.size();

        float32x4_t zero_vec = vdupq_n_f32(0.f);
        float32x4_t factor_vector = vdupq_n_f32(factor);
        for (uint32_t i = 0; i < num_pixels; i += 4)
        {
            float32x4_t points_vec = vld1q_f32(&points[i]);
            uint32x4_t mask = vcgtq_f32(points_vec, zero_vec);
            float32x4_t depth_vec = vdivq_f32(factor_vector, points_vec);
            uint16x4_t depth_int16_vec = vmovn_u32(vcvtq_u32_f32(vbslq_f32(mask, depth_vec, zero_vec)));
            vst1_u16(&depth_data[i], depth_int16_vec);
        }
    }
};

struct Image3D
{
    static void DisparityToDepth(
        std::vector<float> &disparity, cv::Mat &depth,
        int H, int W, float fx, float fy, float cx, float cy, float baseline)
    {
    }
    static void DepthToPointCloud()
    {
    }
    static void DisparityToPointCloud(std::vector<float> &depth, int H, int W)
    {
    }
    static void PointCloudToLaserscan()
    {
    }
};
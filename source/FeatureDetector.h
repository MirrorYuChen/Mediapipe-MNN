/*
 * @Author: chenjingyu
 * @Date: 2023-08-18 19:33:57
 * @LastEditTime: 2023-08-19 22:07:00
 * @Description: Feature Detector
 * @FilePath: \Mediapipe-MNN\source\FeatureDetector.h
 */
#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <MNN/ImageProcess.hpp>
#include <MNN/Interpreter.hpp>
#include <MNN/Matrix.h>
#include <MNN/Tensor.hpp>
#include <memory>
#include <vector>

namespace mirror {
class FeatureDetector {
public:
  FeatureDetector();
  ~FeatureDetector();
  bool LoadModel(const char *model_file);
  void setMaxFeatures(int max_features);
  bool Process(const cv::Mat &input_view, std::vector<cv::KeyPoint> &keypts, std::vector<float> &descriptors);

private:
  void ComputeImagePyramid(const cv::Mat &input_image, std::vector<cv::Mat> *image_pyramid);
  cv::Mat ExtractPatch(const cv::KeyPoint& feature, const std::vector<cv::Mat>& image_pyramid);

private:
  int pyramid_level_ = 4;
  int max_features_ = 200;
  float scale_factor_ = 1.2f;
  int output_width_ = 640;
  int output_height_ = 640;

  bool inited_ = false;
  int input_w_ = 0;
  int input_h_ = 0;
  int input_c_ = 0;
  int input_n_ = 0;
  std::unique_ptr<MNN::Interpreter> net_ = nullptr;
  MNN::Session *sess_ = nullptr;
  MNN::Tensor *input_tensor_ = nullptr;

  cv::Ptr<cv::Feature2D> detector_;
};
} // namespace mirror
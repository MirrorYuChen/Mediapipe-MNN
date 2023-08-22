/*
 * @Author: chenjingyu
 * @Date: 2023-08-21 19:09:53
 * @LastEditTime: 2023-08-22 10:22:21
 * @Description: Segment detector
 * @FilePath: \Mediapipe-MNN\source\SegmentDetector.h
 */
#pragma once

#include "TypeDefines.h"
#include <MNN/ImageProcess.hpp>
#include <MNN/Interpreter.hpp>
#include <MNN/Matrix.h>
#include <MNN/Tensor.hpp>
#include <memory>
#include <vector>
#include <opencv2/core.hpp>

namespace mirror {
class SegmentDetector {
public:
  SegmentDetector() = default;
  ~SegmentDetector();

  bool LoadModel(const char *model_file);
  void setFormat(int format);
  bool Detect(const ImageHead &in, RotateType type, ImageHead &out);

private:
  bool inited_ = false;
  int input_w_ = 0;
  int input_h_ = 0;
  std::shared_ptr<MNN::CV::ImageProcess> pretreat_ = nullptr;
  std::unique_ptr<MNN::Interpreter> net_ = nullptr;
  MNN::Session *sess_ = nullptr;
  MNN::Tensor *input_tensor_ = nullptr;
  MNN::CV::Matrix trans_;
  cv::Mat result_;

  const float meanVals_[3] = {127.5f, 127.5f, 127.5f};
  const float normVals_[3] = {1 / 127.5f, 1 / 127.5f, 1 / 127.5f};
};

} // namespace mirror

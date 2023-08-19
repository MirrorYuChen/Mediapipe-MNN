/*
 * @Author: chenjingyu
 * @Date: 2023-08-19 22:30:03
 * @LastEditTime: 2023-08-19 23:15:21
 * @Description: Pose Detector
 * @FilePath: \Mediapipe-MNN\source\PoseDetector.h
 */
#pragma once

#include "TypeDefines.h"
#include <vector>
#include <memory>
#include <MNN/Matrix.h>
#include <MNN/ImageProcess.hpp>
#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>

namespace mirror {
class PoseDetector {
public:
  PoseDetector() = default;
  ~PoseDetector();

  bool LoadModel(const char *model_file);
  void setFormat(int format);
  bool Detect(const ImageHead &in, RotateType type, std::vector<ObjectInfo> &objects);
  
private:
  void ParseOutputs(MNN::Tensor *bboxes, MNN::Tensor *scores,
                    std::vector<ObjectInfo> &objects);

private:
  bool inited_ = false;
  int input_w_ = 0;
  int input_h_ = 0;
  std::shared_ptr<MNN::CV::ImageProcess> pretreat_ = nullptr;
  std::unique_ptr<MNN::Interpreter> net_ = nullptr;
  MNN::Session *sess_ = nullptr;
  MNN::Tensor *input_tensor_ = nullptr;
  MNN::CV::Matrix trans_;
  float score_thresh_ = 0.5f;
  float iou_thresh_ = 0.5f;

  const float meanVals_[3] = {0.0f, 0.0f, 0.0f};
  const float normVals_[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
};
} // namespace mirror

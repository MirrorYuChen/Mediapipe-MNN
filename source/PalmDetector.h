/*
 * @Author: chenjingyu
 * @Date: 2023-06-19 17:20:56
 * @LastEditTime: 2023-06-21 10:03:06
 * @Description: palm detector
 * @FilePath: \Mediapipe-Hand\source\PalmDetector.h
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
class PalmDetector {
public:
  PalmDetector() = default;
  ~PalmDetector();

  bool LoadModel(const char *model_file);
  void setSourceFormat(int format);
  void setInputSize(int in_w, int in_h, RotateType type);
  bool Detect(const ImageHead &in, RotateType type, std::vector<ObjectInfo> &objects);
  
private:
  void ParseOutputs(MNN::Tensor *scores, MNN::Tensor *boxes, LandmarkList &result);

private:
  bool inited_ = false;
  int input_w_ = 0;
  int input_h_ = 0;
  std::shared_ptr<MNN::CV::ImageProcess> pretreat_ = nullptr;
  std::unique_ptr<MNN::Interpreter> net_ = nullptr;
  MNN::Session *sess_ = nullptr;
  MNN::Tensor *input_tensor_ = nullptr;
  MNN::CV::Matrix trans_;
  float score_thresh_ = 0.6f;

  const float meanVals_[3] = {0.0f, 0.0f, 0.0f};
  const float normVals_[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
  const float nmsThreshold_ = 0.5f;
};

} // namespace mirror

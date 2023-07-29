/*
 * @Author: chenjingyu
 * @Date: 2023-07-29 15:46:48
 * @LastEditTime: 2023-07-29 15:53:11
 * @Description: face detector module
 * @FilePath: \Mediapipe-Hand\source\FaceDetector.h
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
class FaceDetector {
public:
  FaceDetector() = default;
  ~FaceDetector();

  bool LoadModel(const char *model_file);
  void setSourceFormat(int format);
  bool Detect(const ImageHead &in, RotateType type, std::vector<ObjectInfo> &objects);
  
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

  const float meanVals_[3] = {127.5f, 127.5f, 127.5f};
  const float normVals_[3] = {1 / 127.5f, 1 / 127.5f, 1 / 127.5f};
  const float iouThreshold_ = 0.5f;
};



} // namespace mirror

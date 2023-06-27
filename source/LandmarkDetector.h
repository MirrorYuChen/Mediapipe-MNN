/*
 * @Author: chenjingyu
 * @Date: 2023-06-25 11:10:57
 * @LastEditTime: 2023-06-25 17:52:58
 * @Description: landmark detector module
 * @FilePath: \Mediapipe-Hand\source\LandmarkDetector.h
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
class LandmarkerDetector {
public:
  LandmarkerDetector() = default;
  ~LandmarkerDetector();

  bool LoadModel(const char *model_file);
  void setSourceFormat(int format);
  bool Detect(const ImageHead &in, RotateType type, std::vector<ObjectInfo> &objects);

private:
  std::vector<Point2f> getPointRegion(const ImageHead &in, RotateType type,
                                      const ObjectInfo &object);

private:
  bool inited_ = false;
  int input_w_ = 0;
  int input_h_ = 0;
  std::shared_ptr<MNN::CV::ImageProcess> pretreat_ = nullptr;
  std::unique_ptr<MNN::Interpreter> net_ = nullptr;
  MNN::Session *sess_ = nullptr;
  MNN::Tensor *input_tensor_ = nullptr;
  MNN::CV::Matrix trans_;

  const float meanVals_[3] = {0.0f, 0.0f, 0.0f};
  const float normVals_[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
};


} // namespace mirror


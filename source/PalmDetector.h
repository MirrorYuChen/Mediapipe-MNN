/*
 * @Author: chenjingyu
 * @Date: 2023-06-19 17:20:56
 * @LastEditTime: 2023-06-20 12:28:27
 * @Description: palm detector
 * @FilePath: \Mediapipe-Hand\source\PalmDetector.h
 */
#pragma once

#include "TypeDefines.h"
#include <vector>

namespace mirror {
class PalmDetector {
public:
  PalmDetector();
  ~PalmDetector();

  bool LoadModel(const char *model_file);
  int Detect(const ImageHead &in, std::vector<ObjectInfo> &objects);

private:
  int input_w_;
  int input_h_;

  const float meanVals_[3] = {0.0f, 0.0f, 0.0f};
  const float normVals_[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
  const float nmsThreshold_ = 0.5f;
};

} // namespace mirror

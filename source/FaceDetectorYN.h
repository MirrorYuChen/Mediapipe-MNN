/*
 * @Author: chenjingyu
 * @Date: 2023-08-02 12:43:25
 * @LastEditTime: 2023-08-02 12:51:26
 * @Description: face detector yu
 * @FilePath: \Mediapipe-MNN\source\FaceDetectorYN.h
 */
#pragma once

#include "TypeDefines.h"
#include <MNN/ImageProcess.hpp>
#include <MNN/Interpreter.hpp>
#include <MNN/Matrix.h>
#include <MNN/Tensor.hpp>
#include <memory>
#include <vector>

namespace mirror {
class FaceDetectorYN {
public:
  FaceDetectorYN() = default;
  ~FaceDetectorYN();

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
  float iou_thresh_ = 0.5f;

  const std::vector<int> strides_ = { 8, 16, 32};
  const std::vector<std::string> output_names_ = { "cls_8", "cls_16", "cls_32", "obj_8", "obj_16", "obj_32", "bbox_8", "bbox_16", "bbox_32", "kps_8", "kps_16", "kps_32" };
};



} // namespace mirror

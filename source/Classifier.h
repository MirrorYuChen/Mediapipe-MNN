/*
 * @Author: chenjingyu
 * @Date: 2023-08-20 03:32:57
 * @LastEditTime: 2023-08-20 18:13:02
 * @Description: Classifier
 * @FilePath: \Mediapipe-MNN\source\Classifier.h
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
class Classifier {
public:
  Classifier() = default;
  ~Classifier();

  bool LoadModel(const char *model_file);
  void setFormat(int format);
  bool Detect(const ImageHead &in, RotateType type,
              std::vector<ClassifierInfo> &out);

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
  bool use_full_ = false;
  std::string cls_name_;
  std::string reg_name_;
};

} // namespace mirror
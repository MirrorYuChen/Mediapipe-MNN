/*
 * @Author: chenjingyu
 * @Date: 2023-08-20 03:33:02
 * @LastEditTime: 2023-08-20 18:18:43
 * @Description: Classifier module
 * @FilePath: \Mediapipe-MNN\source\Classifier.cc
 */
#include "Classifier.h"
#include "Utils.h"
#include <iostream>

using namespace MNN;
namespace mirror {
Classifier::~Classifier() {
  net_->releaseModel();
  net_->releaseSession(sess_);
}

bool Classifier::LoadModel(const char *model_file) {
  std::cout << "Start load model." << std::endl;
  // 1.load model
  net_ = std::unique_ptr<Interpreter>(Interpreter::createFromFile(model_file));
  if (net_ == nullptr) {
    std::cout << "Failed load model." << std::endl;
    return false;
  }

  // 2.create session
  ScheduleConfig schedule_config;
  schedule_config.type = MNN_FORWARD_CPU;
  schedule_config.numThread = 4;
  sess_ = net_->createSession(schedule_config);
  input_tensor_ = net_->getSessionInput(sess_, nullptr);

  input_h_ = input_tensor_->height();
  input_w_ = input_tensor_->width();

  cls_name_ = "classificators";
  reg_name_ = "regressors";

  std::cout << "End load model." << std::endl;
  inited_ = true;
  return true;
}

void Classifier::setFormat(int format) {
  // create image process
  CV::ImageProcess::Config image_process_config;
  image_process_config.filterType = CV::BILINEAR;
  image_process_config.sourceFormat = CV::ImageFormat(format);
  image_process_config.destFormat = CV::RGB;
  image_process_config.wrap = CV::ZERO;
  memcpy(image_process_config.mean, meanVals_, sizeof(meanVals_));
  memcpy(image_process_config.normal, normVals_, sizeof(normVals_));
  pretreat_ = std::shared_ptr<CV::ImageProcess>(
      CV::ImageProcess::create(image_process_config));
}

bool Classifier::Detect(const ImageHead &in, RotateType type,
                        std::vector<ClassifierInfo> &out) {
  std::cout << "Start detect." << std::endl;
  out.clear();
  if (in.data == nullptr) {
    std::cout << "Error Input empty." << std::endl;
    return false;
  }

  if (!inited_) {
    std::cout << "Model Uninitialized." << std::endl;
    return false;
  }
  // 1.set input
  int width = in.width;
  int height = in.height;
  std::vector<Point2f> input_region = getInputRegion(in, type, input_w_, input_h_);
  
  float points_src[] = {
    input_region[0].x, input_region[0].y, 
    input_region[1].x, input_region[1].y,
    input_region[2].x, input_region[2].y,
    input_region[3].x, input_region[3].y,
  };
  float points_dst[] = {
    0.0f,  0.0f,
    0.0f,  (float)(input_h_ - 1),
    (float)(input_w_ - 1), 0.0f,
    (float)(input_w_ - 1), (float)(input_h_ - 1),
  };
  trans_.setPolyToPoly((CV::Point *)points_dst, (CV::Point *)points_src, 4);
  pretreat_->setMatrix(trans_);
  pretreat_->convert((uint8_t *)in.data, width, height, in.width_step, input_tensor_);

  // 2.do inference
  int ret = net_->runSession(sess_);
  if (ret != 0) {
    std::cout << "Failed do inference." << std::endl;
    return -1;
  }

  // 3.get the result
  MNN::Tensor *classifier = net_->getSessionOutput(sess_, nullptr);
  if (nullptr == classifier) {
    std::cout << "Error output." << std::endl;
    return false;
  }
  std::shared_ptr<MNN::Tensor> output_classifier(new MNN::Tensor(classifier, classifier->getDimensionType()));
  classifier->copyToHostTensor(output_classifier.get());

  float *ptr = output_classifier->host<float>();
  out.resize(output_classifier->channel());
  ClassifierInfo info;
  for (int k = 0; k < output_classifier->channel(); ++k) {
    out[k].id = k;
    out[k].score = ptr[k];
  }
  return true;
}

} // namespace mirror

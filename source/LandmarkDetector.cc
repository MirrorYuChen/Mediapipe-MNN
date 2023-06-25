/*
 * @Author: chenjingyu
 * @Date: 2023-06-25 11:11:06
 * @LastEditTime: 2023-06-25 12:31:30
 * @Description: landmark detector module
 * @FilePath: \Mediapipe-Hand\source\LandmarkDetector.cc
 */
#include "LandmarkDetector.h"
#include <iostream>
#include "Utils.h"

namespace mirror {
using namespace MNN;
LandmarkerDetector::~LandmarkerDetector() {
  net_->releaseModel();
  net_->releaseSession(sess_);
}

bool LandmarkerDetector::LoadModel(const char *model_file) {
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

  std::cout << "End load model." << std::endl;
  inited_ = true;
  return true;
}

void LandmarkerDetector::setSourceFormat(int format) {
  // create image process
  CV::ImageProcess::Config image_process_config;
  image_process_config.filterType = CV::BILINEAR;
  image_process_config.sourceFormat = CV::ImageFormat(format);
  image_process_config.destFormat = CV::RGB;
  image_process_config.wrap = CV::ZERO;
  memcpy(image_process_config.mean, meanVals_, sizeof(meanVals_));
  memcpy(image_process_config.normal, normVals_, sizeof(normVals_));
  pretreat_ = std::shared_ptr<CV::ImageProcess>(CV::ImageProcess::create(image_process_config));
}

void LandmarkerDetector::setInputSize(int in_w, int in_h, RotateType type) {
  std::vector<Point2f> input_region = getInputRegion(in_w, in_h, input_w_, input_h_, type);
  float points_src[] = {
    input_region[0].x, input_region[0].y,
    input_region[1].x, input_region[1].y,
    input_region[2].x, input_region[2].y,
    input_region[3].x, input_region[3].y,
  };
  float points_dst[] = {
    0.0f, 0.0f,
    0.0f, (float)(input_h_ - 1),
    (float)(input_w_ - 1), 0.0f,
    (float)(input_w_ - 1), (float)(input_h_ - 1),
  };
  trans_.setPolyToPoly((CV::Point*)points_dst, (CV::Point*)points_src, 4);
  pretreat_->setMatrix(trans_);
}

bool LandmarkerDetector::Detect(const ImageHead &in, RotateType type, std::vector<Point2f> &landmarks) {
  std::cout << "Start detect." << std::endl;
  landmarks.clear();
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
}


} // namespace mirror

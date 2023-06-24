/*
 * @Author: chenjingyu
 * @Date: 2023-06-19 17:37:42
 * @LastEditTime: 2023-06-21 10:07:11
 * @Description: palm detector module
 * @FilePath: \Mediapipe-Hand\source\PalmDetector.cc
 */
#include "PalmDetector.h"
#include <iostream>
#include "Utils.h"

namespace mirror {
using namespace MNN;
PalmDetector::~PalmDetector() {
  net_->releaseModel();
  net_->releaseSession(sess_);
}

bool PalmDetector::LoadModel(const char *model_file) {
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

void PalmDetector::setSourceFormat(int format) {
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

void PalmDetector::setInputSize(int in_w, int in_h, RotateType type) {
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

bool PalmDetector::Detect(const ImageHead &in, RotateType type,
                         std::vector<ObjectInfo> &objects) {
  std::cout << "Start detect." << std::endl;
  objects.clear();
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
  pretreat_->convert((uint8_t*)in.data, width, height, in.width_step, input_tensor_);
 
  // 2.do inference
  int ret = net_->runSession(sess_);
  if (ret != 0) {
    std::cout << "Failed do inference." << std::endl;
    return -1;
  }
  
  // 3.get the result
  MNN::Tensor *classify = net_->getSessionOutput(sess_, "classificators");
  MNN::Tensor *regressor = net_->getSessionOutput(sess_, "regressors");
  if (nullptr == classify || nullptr == regressor) {
    std::cout << "Error output." << std::endl;
    return false;
  }
  std::shared_ptr<MNN::Tensor> output_classify(new MNN::Tensor(classify, classify->getDimensionType()));
  std::shared_ptr<MNN::Tensor> output_regressor(new MNN::Tensor(regressor, regressor->getDimensionType()));
  classify->copyToHostTensor(output_classify.get());
  regressor->copyToHostTensor(output_regressor.get());
  
  // 4.parse the result
  printf("classify nchw: %d x %d x %d x %d.\n", output_classify->batch(), output_classify->channel(), output_classify->height(), output_classify->width()); 
  printf("regression nchw: %d x %d x %d x %d.\n", output_regressor->batch(), output_regressor->channel(), output_regressor->height(), output_regressor->width());


  std::cout << "End detect." << std::endl;
  return true;
}

void PalmDetector::ParseOutputs(MNN::Tensor *scores, MNN::Tensor *boxes, LandmarkList &result) {
  result.clear();
  float *scores_ptr = scores->host<float>();
  float *boxes_ptr = boxes->host<float>();

  int channel = scores->channel();
  
}

} // namespace mirror

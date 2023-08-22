/*
 * @Author: chenjingyu
 * @Date: 2023-08-21 19:10:01
 * @LastEditTime: 2023-08-22 10:30:50
 * @Description: Segment detector
 * @FilePath: \Mediapipe-MNN\source\SegmentDetector.cc
 */
#include "SegmentDetector.h"
#include "Utils.h"
#include "CommonData.h"
#include <iostream>

using namespace MNN;
namespace mirror {
SegmentDetector::~SegmentDetector() {
  net_->releaseModel();
  net_->releaseSession(sess_);
}

bool SegmentDetector::LoadModel(const char *model_file) {
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

void SegmentDetector::setFormat(int format) {
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

bool SegmentDetector::Detect(const ImageHead &in, RotateType type, ImageHead &out) {
  std::cout << "Start detect." << std::endl;
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
  std::vector<Point2f> input_region =
      getInputRegion(in, type, input_w_, input_h_);

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
  MNN::Tensor *result = net_->getSessionOutput(sess_, nullptr);
  if (nullptr == result) {
    std::cout << "Error output." << std::endl;
    return false;
  }
  std::shared_ptr<MNN::Tensor> output_result(new MNN::Tensor(result, result->getDimensionType()));
  result->copyToHostTensor(output_result.get());

  float *ptr = output_result->host<float>();
  int output_height = output_result->channel();
  int output_width = output_result->height();
  int depth = output_result->width();
  const int height_step = output_width * depth;
  cv::Mat dst_image = cv::Mat::zeros(output_height, output_width, CV_8UC3);
  for (int h = 0; h < output_height; ++h) {
    for (int w = 0; w < output_width; ++w) {
      int class_index = 0;
      float max_confidence = 0.0f;
      for (int d = 0; d < depth; ++d) {
        float confidence = ptr[h * height_step + w * depth + d];
        if (confidence > max_confidence) {
          max_confidence = confidence;
          class_index = d;
        }
      }
      dst_image.at<cv::Vec3b>(h, w)[0] = kColorMap[3 * class_index + 0];
      dst_image.at<cv::Vec3b>(h, w)[1] = kColorMap[3 * class_index + 1];
      dst_image.at<cv::Vec3b>(h, w)[2] = kColorMap[3 * class_index + 2];
    }
  }

  float pt_dst[] = {
    input_region[0].x, input_region[0].y, 
    input_region[1].x, input_region[1].y,
    input_region[2].x, input_region[2].y,
    input_region[3].x, input_region[3].y,
  };
  float pt_src[] = {
    0.0f, 0.0f,
    0.0f, (float)(output_height - 1),
    (float)(output_width - 1), 0.0f,
    (float)(output_width - 1), (float)(output_height - 1),
  };
  trans_.setPolyToPoly((CV::Point *)pt_dst, (CV::Point *)pt_src, 4);
  CV::ImageProcess::Config image_process_config;
  image_process_config.filterType = CV::NEAREST;
  image_process_config.sourceFormat = CV::RGB;
  image_process_config.destFormat = CV::BGR;
  image_process_config.wrap = CV::ZERO;
  std::shared_ptr<CV::ImageProcess> pretreat = std::shared_ptr<CV::ImageProcess>(CV::ImageProcess::create(image_process_config));
  pretreat->setMatrix(trans_);

  result_ = cv::Mat(in.height, in.width, CV_8UC3);
  std::shared_ptr<MNN::Tensor> dst_tensor(MNN::Tensor::create<uint8_t>(std::vector<int>{1, result_.rows, result_.cols, result_.channels()}, result_.data));
  pretreat->convert((uint8_t *)dst_image.data, dst_image.cols, dst_image.rows, dst_image.step[0], dst_tensor.get());

  out.data = result_.data;
  out.width = result_.cols;
  out.height = result_.rows;
  out.width_step = result_.step[0];
  out.time_stamp = 0.0;
  out.pixel_format = BGR;
  return true;
}

} // namespace mirror

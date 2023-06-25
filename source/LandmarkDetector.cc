/*
 * @Author: chenjingyu
 * @Date: 2023-06-25 11:11:06
 * @LastEditTime: 2023-06-25 17:27:07
 * @Description: landmark detector module
 * @FilePath: \Mediapipe-Hand\source\LandmarkDetector.cc
 */
#include "LandmarkDetector.h"
#include <iostream>
#include <cfloat>
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

bool LandmarkerDetector::Detect(const ImageHead &in, RotateType type, std::vector<ObjectInfo> &objects) {
  std::cout << "Start detect." << std::endl;
  if (in.data == nullptr) {
    std::cout << "Error Input empty." << std::endl;
    return false;
  }

  if (!inited_) {
    std::cout << "Model Uninitialized." << std::endl;
    return false;
  }
  int width = in.width;
  int height = in.height;
  for (auto &object : objects) {
    float angle = object.rotation;
    switch (type) {
    case RotateType::CLOCKWISE_ROTATE_90:
      angle += 90.0f;
      break;
    case RotateType::CLOCKWISE_ROTATE_180:
      angle += 180.0f;
      break;
    case RotateType::CLOCKWISE_ROTATE_270:
      angle += 270.0f;
      break;    
    default:
      break;
    }
    // 1.1 rotate region
    CV::Matrix trans;
    trans.postRotate(angle, 0.5f * width, 0.5f * height);
    std::vector<Point2f> region(4), rotated_region(4);
    Point2f center, rotated_center;
    region[0] = object.tl, region[3] = object.br;
    region[1].x = region[0].x, region[1].y = region[3].y;
    region[2].x = region[2].x, region[2].y = region[0].y;
    center.x = 0.5f * (region[0].x + region[3].x);
    center.y = 0.5f * (region[0].y + region[3].y);
    float xmin = FLT_MAX;
    float ymin = FLT_MAX;
    float xmax = FLT_MIN;
    float ymax = FLT_MIN;
    rotated_center.x = trans[0] * center.x + trans[1] * center.y + trans[2];
    rotated_center.y = trans[3] * center.x + trans[4] * center.y + trans[5];
    for (int i = 0; i < 4; ++i) {
      rotated_region[i].x = trans[0] * region[i].x + trans[1] * region[i].y + trans[2];
      rotated_region[i].y = trans[3] * region[i].x + trans[4] * region[i].y + trans[5];
      xmin = MIN_(rotated_region[i].x, xmin);
      ymin = MIN_(rotated_region[i].y, ymin);
      xmax = MAX_(rotated_region[i].x, xmax);
      ymax = MAX_(rotated_region[i].y, ymax);
    }

    // 1.2 get width and height;
    float region_width = xmax - xmin;
    float region_height = ymax - ymin;
    float max_side = 2.6f * MAX_(region_width, region_height);
    region[0].x = rotated_center.x - 0.5f * region_width;
    region[0].y = rotated_center.y - 0.5f * region_height;
    region[1].x = rotated_center.x - 0.5f * region_width;
    region[1].y = rotated_center.y + 0.5f * region_height;
    region[2].x = rotated_center.x + 0.5f * region_width;
    region[2].y = rotated_center.y - 0.5f * region_height;
    region[3].x = rotated_center.x + 0.5f * region_width;
    region[3].y = rotated_center.y + 0.5f * region_height;

    // 1.3 get the origin region
    trans.invert(&trans);
    for (int i = 0; i < 4; ++i) {
      rotated_region[i].x = trans[0] * region[i].x + trans[1] * region[i].y + trans[2];
      rotated_region[i].y = trans[3] * region[i].x + trans[4] * region[i].y + trans[5];
    }

    float points_src[] = {
      rotated_region[0].x, rotated_region[0].y,
      rotated_region[1].x, rotated_region[1].y,
      rotated_region[2].x, rotated_region[2].y,
      rotated_region[3].x, rotated_region[3].y,
    };
    float points_dst[] = {
      0.0f, 0.0f,
      0.0f, (float)(input_h_ - 1),
      (float)(input_w_ - 1), 0.0f,
      (float)(input_w_ - 1), (float)(input_h_ - 1),
    };
    trans_.setPolyToPoly((CV::Point*)points_dst, (CV::Point*)points_src, 4);
    pretreat_->setMatrix(trans_);

    // 1.4 set the input
    pretreat_->convert((uint8_t *)in.data, width, height, in.width_step, input_tensor_);

    // 1.5.do inference
    int ret = net_->runSession(sess_);
    if (ret != 0) {
      std::cout << "Failed do inference." << std::endl;
      return -1;
    }

    // 1.6 get the result
    MNN::Tensor *palm_score = net_->getSessionOutput(sess_, "Identity_1");
    MNN::Tensor *left_right = net_->getSessionOutput(sess_, "Identity_2");
    MNN::Tensor *landmark_norm = net_->getSessionOutput(sess_, "Identity");
    MNN::Tensor *landmark_world = net_->getSessionOutput(sess_, "Identity_3");

    std::shared_ptr<MNN::Tensor> output_left_right(
      new MNN::Tensor(left_right, left_right->getDimensionType()));
    std::shared_ptr<MNN::Tensor> output_palm_score(
      new MNN::Tensor(palm_score, palm_score->getDimensionType()));
    std::shared_ptr<MNN::Tensor> output_landmark_norm(
      new MNN::Tensor(landmark_norm, landmark_norm->getDimensionType()));
    std::shared_ptr<MNN::Tensor> output_landmark_world(
      new MNN::Tensor(landmark_world, landmark_world->getDimensionType()));

    left_right->copyToHostTensor(output_left_right.get());
    palm_score->copyToHostTensor(output_palm_score.get());
    landmark_norm->copyToHostTensor(output_landmark_norm.get());
    landmark_world->copyToHostTensor(output_landmark_world.get());

    float *left_right_ptr = output_left_right->host<float>();
    if (left_right_ptr[0] > 0.5f) {
      object.left_right = 1;
    } else {
      object.left_right = 0;
    }

    std::vector<Point2f> landmarks(21);
    for (int i = 0; i < 21; ++i) {
      float *landmarks_ptr = landmark_norm->host<float>();
      landmarks[i].x = landmarks_ptr[3 * i + 0];
      landmarks[i].y = landmarks_ptr[3 * i + 1];
      object.landmarks[i].x =
          trans_[0] * landmarks[i].x + trans_[1] * landmarks[i].y + trans_[2];
      object.landmarks[i].y =
          trans_[3] * landmarks[i].x + trans_[4] * landmarks[i].y + trans_[5];
    }


    //printf("left right: nchw: %d, %d, %d, %d.\n", output_left_right->batch(), output_left_right->channel(), output_left_right->height(), output_left_right->width());
    //printf("palm score: nchw: %d, %d, %d, %d.\n", output_palm_score->batch(), output_palm_score->channel(), output_palm_score->height(), output_palm_score->width());
    //printf("output landmark normalize: nchw: %d, %d, %d, %d.\n", output_landmark_norm->batch(), output_landmark_norm->channel(), output_landmark_norm->height(), output_landmark_norm->width());
    //printf("output landmark: nchw: %d, %d, %d, %d.\n", output_landmark_world->batch(), output_landmark_world->channel(), output_landmark_world->height(), output_landmark_world->width());
  }


}


} // namespace mirror

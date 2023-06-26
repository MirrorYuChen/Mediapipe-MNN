/*
 * @Author: chenjingyu
 * @Date: 2023-06-25 11:11:06
 * @LastEditTime: 2023-06-26 12:42:16
 * @Description: landmark detector module
 * @FilePath: \Mediapipe-Hand\source\LandmarkDetector.cc
 */
#include "LandmarkDetector.h"
#include "Utils.h"
#include <cfloat>
#include <iostream>

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
  pretreat_ = std::shared_ptr<CV::ImageProcess>(
      CV::ImageProcess::create(image_process_config));
}

float LandmarkerDetector::getAlignAngle(const ImageHead &in, RotateType type,
                                        const ObjectInfo &object) {
  int width = in.width;
  int height = in.height;
  Point2f src = object.index_landmarks[0];
  Point2f dst = object.index_landmarks[2];
  CV::Matrix trans;
  trans.preRotate(RotateTypeToAngle(type), 0.5f * width, 0.5f * height);
  Point2f src_align, dst_align;
  src_align.x = trans[0] * src.x + trans[1] * src.y + trans[2];
  src_align.y = trans[3] * src.x + trans[4] * src.y + trans[5];

  dst_align.x = trans[0] * dst.x + trans[1] * dst.y + trans[2];
  dst_align.y = trans[3] * dst.x + trans[4] * dst.y + trans[5];

  float dx = dst_align.x - src_align.x;
  float dy = dst_align.y - src_align.y;
  return -(90.0f - std::atan2(-dy, dx) * 180.0f / M_PI);
}

std::vector<Point2f>
LandmarkerDetector::getPointRegion(const ImageHead &in, RotateType type,
                                   const ObjectInfo &object) {
  int width = in.width;
  int height = in.height;
  // 1.align the image
  float init_angle = getAlignAngle(in, type, object);
  float angle = RotateTypeToAngle(type) + init_angle;

  // 2.get the align region
  CV::Matrix trans;
  trans.postRotate(angle, 0.5f * width, 0.5f * height);
  float rect_width = object.br.x - object.tl.x;
  float rect_height = object.br.y - object.tl.y;

  Point2f center;
  center.x = 0.5f * (object.br.x + object.tl.x);
  center.y = 0.5f * (object.br.y + object.tl.y);
  float center_x = trans[0] * center.x + trans[1] * center.y + trans[2];
  float center_y = trans[3] * center.x + trans[4] * center.y + trans[5] - 0.5f * rect_height;

  // 3. expand the region
  float half_max_side = MAX_(rect_width, rect_height) * 1.3f;
  float xmin = center_x - half_max_side;
  float ymin = center_y - half_max_side;
  float xmax = center_x + half_max_side;
  float ymax = center_y + half_max_side;

  std::vector<Point2f> region(4);
  region[0].x = xmin;
  region[0].y = ymin;
  region[1].x = xmin;
  region[1].y = ymax;
  region[2].x = xmax;
  region[2].y = ymin;
  region[3].x = xmax;
  region[3].y = ymax;

  std::vector<Point2f> result(4);
  trans.invert(&trans);
  for (size_t i = 0; i < region.size(); ++i) {
    result[i].x = trans[0] * region[i].x + trans[1] * region[i].y + trans[2];
    result[i].y = trans[3] * region[i].x + trans[4] * region[i].y + trans[5];
  }
  return result;
}

bool LandmarkerDetector::Detect(const ImageHead &in, RotateType type,
                                std::vector<ObjectInfo> &objects) {
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
    std::vector<Point2f> region = getPointRegion(in, type, object);
    float points_src[] = {
      region[0].x, region[0].y, region[1].x, region[1].y,
      region[2].x, region[2].y, region[3].x, region[3].y,
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

    // printf("left right: nchw: %d, %d, %d, %d.\n", output_left_right->batch(),
    // output_left_right->channel(), output_left_right->height(),
    // output_left_right->width()); printf("palm score: nchw: %d, %d, %d,
    // %d.\n", output_palm_score->batch(), output_palm_score->channel(),
    // output_palm_score->height(), output_palm_score->width()); printf("output
    // landmark normalize: nchw: %d, %d, %d, %d.\n",
    // output_landmark_norm->batch(), output_landmark_norm->channel(),
    // output_landmark_norm->height(), output_landmark_norm->width());
    // printf("output landmark: nchw: %d, %d, %d, %d.\n",
    // output_landmark_world->batch(), output_landmark_world->channel(),
    // output_landmark_world->height(), output_landmark_world->width());
  }

  return true;
}

} // namespace mirror

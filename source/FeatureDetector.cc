#include "FeatureDetector.h"
#include "TypeDefines.h"
#include <cmath>

using namespace MNN;
namespace mirror {
constexpr int kPatchSize = 32;
FeatureDetector::FeatureDetector() {
  detector_ = cv::ORB::create(max_features_, scale_factor_, pyramid_level_,
                              kPatchSize - 1, 0, 2, cv::ORB::FAST_SCORE);
}

FeatureDetector::~FeatureDetector() {
  net_->releaseModel();
  net_->releaseSession(sess_);
}

bool FeatureDetector::LoadModel(const char *model_file) {
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

  input_n_ = input_tensor_->batch();
  input_c_ = input_tensor_->channel();
  input_h_ = input_tensor_->height();
  input_w_ = input_tensor_->width();

  std::cout << "End load model." << std::endl;
  inited_ = true;
  return true;
}

void FeatureDetector::setMaxFeatures(int max_features) {
  max_features_ = max_features;
  detector_ = cv::ORB::create(max_features_, scale_factor_, pyramid_level_,
                              kPatchSize - 1, 0, 2, cv::ORB::FAST_SCORE);
}

bool FeatureDetector::Process(const cv::Mat &input_view,
                              std::vector<cv::KeyPoint> &keypts,
                              std::vector<float> &descriptors) {
  if (input_view.empty()) {
    std::cout << "Empty input." << std::endl;
    return false;
  }
  keypts.clear();
  descriptors.clear();
  // resize: 640 x 640
  cv::Mat scaled_mat;
  int input_width = input_view.cols;
  int input_height = input_view.rows;
  const float scale =
      std::min(static_cast<float>(output_width_) / input_width,
               static_cast<float>(output_height_) / input_height);
  const int target_width = std::round(input_width * scale);
  const int target_height = std::round(input_height * scale);
  int scale_flag = scale < 1.0f ? cv::INTER_AREA : cv::INTER_LINEAR;
  cv::resize(input_view, scaled_mat, cv::Size(target_width, target_height), 0, 0, scale_flag);
  output_width_ = target_width;
  output_height_ = target_height;

  cv::Mat grayscale_view;
  cv::cvtColor(scaled_mat, grayscale_view, cv::COLOR_BGR2GRAY);
  detector_->detect(grayscale_view, keypts);
  if (keypts.size() > max_features_) {
    keypts.resize(max_features_);
  }
  for (auto &keypt : keypts) {
    keypt.pt.x /= scale;
    keypt.pt.y /= scale;
  }
  std::vector<cv::Mat> image_pyramid;
  ComputeImagePyramid(grayscale_view, &image_pyramid);
  std::vector<cv::Mat> patch_mat;
  patch_mat.resize(keypts.size());
  for (int i = 0; i < keypts.size(); ++i) {
    patch_mat[i] = ExtractPatch(keypts[i], image_pyramid);
  }

  // PATCHES
  const int batch_size = max_features_;
  int num_bytes = batch_size * kPatchSize * kPatchSize * sizeof(float);
  float *tensor_buffer = input_tensor_->host<float>();
  for (int i = 0; i < keypts.size(); i++) {
    for (int j = 0; j < patch_mat[i].rows; ++j) {
      for (int k = 0; k < patch_mat[i].cols; ++k) {
        *tensor_buffer++ = patch_mat[i].at<uchar>(j, k) / 128.0f - 1.0f;
      }
    }
  }
  for (int i = keypts.size() * kPatchSize * kPatchSize; i < num_bytes / 4; i++) {
    *tensor_buffer++ = 0;
  }

  // 2.do inference
  int ret = net_->runSession(sess_);
  if (ret != 0) {
    std::cout << "Failed do inference." << std::endl;
    return -1;
  }

  // 3.get the extracted descriptors: knift_feature_floats
  MNN::Tensor *output = net_->getSessionOutput(sess_, nullptr);
  std::shared_ptr<MNN::Tensor> result(new MNN::Tensor(output, output->getDimensionType()));
  output->copyToHostTensor(result.get());

  int descriptors_dims =
      result->batch() * result->channel() * result->height() * result->width();
  descriptors.resize(descriptors_dims);
  memcpy(descriptors.data(), output->host<float>(), descriptors_dims * sizeof(float));

  return true;
}

void FeatureDetector::ComputeImagePyramid(const cv::Mat &input_image, std::vector<cv::Mat> *image_pyramid) {
  cv::Mat tmp_image = input_image;
  cv::Mat src_image = input_image;
  for (int i = 0; i < pyramid_level_; ++i) {
    image_pyramid->push_back(src_image);
    cv::resize(src_image, tmp_image, cv::Size(), 1.0f / scale_factor_,
               1.0f / scale_factor_);
    src_image = tmp_image;
  }
}

cv::Mat
FeatureDetector::ExtractPatch(const cv::KeyPoint &feature, const std::vector<cv::Mat> &image_pyramid) {
  cv::Mat img = image_pyramid[feature.octave];
  float scale_factor = 1 / pow(scale_factor_, feature.octave);
  cv::Point2f center =
      cv::Point2f(feature.pt.x * scale_factor, feature.pt.y * scale_factor);
  cv::Mat rot = cv::getRotationMatrix2D(center, feature.angle, 1.0);
  rot.at<double>(0, 2) += kPatchSize / 2 - center.x;
  rot.at<double>(1, 2) += kPatchSize / 2 - center.y;
  cv::Mat cropped_img;
  // perform the affine transformation
  cv::warpAffine(img, cropped_img, rot, cv::Size(kPatchSize, kPatchSize),
                 cv::INTER_LINEAR);
  return cropped_img;
}

} // namespace mirror

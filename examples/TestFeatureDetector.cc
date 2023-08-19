#include <opencv2/opencv.hpp>
#include "FeatureDetector.h"
#include <memory>
#include <iostream>

using namespace mirror;
constexpr int DIMS = 40;
cv::Mat toDescriptor(const std::vector<float> &data) {
  const int feature_size = data.size() / DIMS;
  cv::Mat descriptor(feature_size, DIMS, CV_32F);
   for (int i = 0; i < feature_size; ++i) {
     for (int j = 0; j < DIMS; ++j) {
       descriptor.at<float>(i, j) = data[i * DIMS + j];
     }
   }
   return descriptor;
}

int main(int argc, char *argv[]) {
  const char *model_file = "../data/models/knift_float_fp16.mnn";
  const char *src_image_file = "../data/images/query.jpg";
  const char *dst_image_file = "../data/images/pattern.jpg";
  int max_features = 200;   // 200 / 400 / 1000
  std::unique_ptr<FeatureDetector> detector = nullptr;
  detector.reset(new FeatureDetector());

  if (!detector->LoadModel(model_file)) {
    std::cout << "failed load model." << std::endl;
    return -1;
  }
  detector->setMaxFeatures(max_features);
  cv::Mat src_image = cv::imread(src_image_file, 1);
  cv::Mat dst_image = cv::imread(dst_image_file, 1);
  if (src_image.empty() || dst_image.empty()) {
    std::cout << "failed load image." << std::endl;
    return -1;
  }

  std::vector<cv::KeyPoint> src_keypts, dst_keypts;
  std::vector<float> src_descs, dst_descs;

  if (!detector->Process(src_image, src_keypts, src_descs) ||
      !detector->Process(dst_image, dst_keypts, dst_descs)) { 
    std::cout << "Failed Process image." << std::endl;
    return -1;
  }
  cv::Mat cv_src_descs = toDescriptor(src_descs);
  cv::Mat cv_dst_descs = toDescriptor(dst_descs);

  cv::BFMatcher bf_matcher(cv::NORM_L2, true);
  int knn = 1;
  std::vector<std::vector<cv::DMatch>> matches;
  bf_matcher.knnMatch(cv_src_descs, cv_dst_descs, matches, knn);
  std::vector<cv::DMatch> good_matches;
  for (const auto &match_pair : matches) {
    if (match_pair.size() < knn) continue;
    const cv::DMatch &best_match = match_pair[0];
    if (best_match.distance > 0.9f) continue;
    good_matches.emplace_back(best_match);
  }
  cv::Mat src_result, dst_result;
  cv::drawKeypoints(src_image, src_keypts, src_result);
  cv::drawKeypoints(dst_image, dst_keypts, dst_result);
  cv::imshow("src_result", src_result);
  cv::imshow("dst_result", dst_result);
  cv::waitKey(0);

  cv::Mat result;
  cv::drawMatches(src_image, src_keypts, dst_image, dst_keypts, good_matches, result);
  cv::imshow("result", result);
  cv::waitKey(0);
  cv::imwrite("../data/results/knift.jpg", result);

  return 0;
}
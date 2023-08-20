/*
 * @Author: chenjingyu
 * @Date: 2023-08-20 18:19:00
 * @LastEditTime: 2023-08-20 19:28:00
 * @Description: Test Classifier module
 * @FilePath: \Mediapipe-MNN\examples\TestClassifier.cc
 */
#include "Classifier.h"
#include "TypeDefines.h"
#include <memory>
#include <algorithm>
#include <vector>
#include <string>
#include <fstream>
#include <vector>
#include <opencv2/opencv.hpp>

static std::vector<std::string> loadLabels(const char *label_file) {
  std::ifstream in(label_file);
  std::vector<std::string> result;
  if (!in.is_open()) {
    return std::vector<std::string>();
  }
  std::string line;
  while (getline(in, line)) {
    result.emplace_back(line);
  }
  in.close();
  return result;
}

using namespace mirror;
int main(int argc, char *argv[]) {
  const char *model_file = "../data/models/classifier_fp16.mnn";
  const char *label_file = "../data/classifier_labels.txt";
  const char *image_file = "../data/images/burger.jpg";
  std::vector<std::string> labels = loadLabels(label_file);
  if (labels.size() == 0) {
    std::cout << "Faile load labels." << std::endl;
    return -1;
  }
  
  std::unique_ptr<Classifier> classifier = nullptr;
  const int topk = 1;
  classifier.reset(new Classifier());

  if (!classifier->LoadModel(model_file)) {
    std::cout << "failed load model: " << model_file << std::endl;
    return -1;
  }
  classifier->setFormat(BGR);

  cv::Mat frame = cv::imread(image_file, 1);
  if (frame.empty()) {
    std::cout << "Failed load image file." << std::endl;
    return -1;
  }
  RotateType type = RotateType::CLOCKWISE_ROTATE_0;
  if (type == CLOCKWISE_ROTATE_90) {
    cv::transpose(frame, frame);
  } else if (type == CLOCKWISE_ROTATE_180) {
    cv::flip(frame, frame, -1);
  }
  if (type == CLOCKWISE_ROTATE_270) {
    cv::transpose(frame, frame);
    cv::flip(frame, frame, 1);
  }

  ImageHead in;
  in.data = frame.data;
  in.height = frame.rows;
  in.width = frame.cols;
  in.width_step = frame.step[0];
  in.pixel_format = PixelFormat::BGR;

  std::vector<ClassifierInfo> result;
  classifier->Detect(in, type, result);
  std::partial_sort(result.begin(), result.begin() + topk, result.end(), [](const ClassifierInfo &a, const ClassifierInfo &b) {
    return a.score > b.score;
  });

  for (int i = 0; i < topk; ++i) {
    cv::putText(frame, labels[result[i].id + 1], cv::Point(10, 10 + 30 * i),
			0, 0.5, cv::Scalar(255, 100, 0), 2, 2);
  }
  cv::imshow("result", frame);
	cv::waitKey(0);

  cv::imwrite("../data/results/mediapipe_classifier_result.jpg", frame);


  return 0;
}

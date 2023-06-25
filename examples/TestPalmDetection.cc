/*
 * @Author: chenjingyu
 * @Date: 2023-06-20 15:43:27
 * @LastEditTime: 2023-06-21 11:47:16
 * @Description: test palm detection
 * @FilePath: \Mediapipe-Hand\examples\TestPalmDetection.cc
 */
#include "TypeDefines.h"
#include "PalmDetector.h"
#include <opencv2/opencv.hpp>

using namespace mirror;

int main(int argc, char *argv[]) {
  const char *image_file = "../data/images/hand.jpg";
  cv::Mat image = cv::imread(image_file);
  if (image.empty()) {
    std::cout << "failed load image." << std::endl;
    return -1;
  }
  ImageHead in;
  in.data = image.data;
  in.height = image.rows;
  in.width = image.cols;
  in.width_step = image.step[0];
  in.pixel_format = PixelFormat::BGR;

  const char *model_file = "../data/models/palm_detection.mnn";
  RotateType type = RotateType::CLOCKWISE_ROTATE_0;
  PalmDetector detector;
  if (!detector.LoadModel(model_file)) {
    std::cout << "Failed load model." << std::endl;
    return -1;
  }
  detector.setSourceFormat(in.pixel_format);

  std::vector<ObjectInfo> objects;
  detector.Detect(in, type, objects);
  for (const auto &object : objects) {
    cv::rectangle(image, cv::Point2f(object.tl.x, object.tl.y),
                  cv::Point2f(object.br.x, object.br.y),
                  cv::Scalar(255, 0, 255), 2);
  }
  cv::imshow("result", image);
  cv::waitKey(0);

  return 0;
}

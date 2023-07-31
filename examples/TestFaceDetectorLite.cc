/*
 * @Author: chenjingyu
 * @Date: 2023-07-30 21:43:30
 * @LastEditTime: 2023-07-31 09:27:47
 * @Description: Test MNN face detector
 * @FilePath: \Mediapipe-MNN\examples\TestFaceDetectorLite.cc
 */
#include "TypeDefines.h"
#include "FaceDetectorLite.h"
#include <opencv2/opencv.hpp>
#include <cmath>

using namespace mirror;

int main(int argc, char *argv[]) {
  const char *image_file = "../data/images/face_demo.jpg";
  cv::Mat image = cv::imread(image_file);
  if (image.empty()) {
    std::cout << "failed load image." << std::endl;
    return -1;
  }
  RotateType type = RotateType::CLOCKWISE_ROTATE_0;
  if (type == CLOCKWISE_ROTATE_90) {
    cv::transpose(image, image);
  } else if (type == CLOCKWISE_ROTATE_180) {
    cv::flip(image, image, -1);
  }
  if (type == CLOCKWISE_ROTATE_270) {
    cv::transpose(image, image);
    cv::flip(image, image, 1);
  }
  ImageHead in;
  in.data = image.data;
  in.height = image.rows;
  in.width = image.cols;
  in.width_step = image.step[0];
  in.pixel_format = PixelFormat::BGR;

  const char *face_model_file = "../data/models/face_detection_lite.mnn";
  FaceDetectorLite detector;
  if (!detector.LoadModel(face_model_file)) {
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
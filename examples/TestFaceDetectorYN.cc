/*
 * @Author: chenjingyu
 * @Date: 2023-08-02 13:17:14
 * @LastEditTime: 2023-08-02 13:19:10
 * @Description: Test Face Detector YN
 * @FilePath: \Mediapipe-MNN\examples\TestFaceDetectorYN.cc
 */
#include "TypeDefines.h"
#include "FaceDetectorYN.h"
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

  const char *face_model_file = "../data/models/face_detection_yunet_2023mar_fp16.mnn";
  FaceDetectorYN detector;
  if (!detector.LoadModel(face_model_file)) {
    std::cout << "Failed load model." << std::endl;
    return -1;
  }
  detector.setFormat(in.pixel_format);
  std::vector<ObjectInfo> objects;
  detector.Detect(in, type, objects);
  for (const auto &object : objects) {
    cv::rectangle(image, cv::Point2f(object.rect.left, object.rect.top),
                  cv::Point2f(object.rect.right, object.rect.bottom),
                  cv::Scalar(255, 0, 255), 2);
    for (int i = 0; i < object.landmarks.size(); ++i) {
      cv::Point pt = cv::Point((int)object.landmarks[i].x, (int)object.landmarks[i].y);
      cv::circle(image, pt, 2, cv::Scalar(255, 255, 0));
    }
  }
  cv::imshow("result", image);
  cv::waitKey(0); 
  cv::imwrite("../data/results/YN_face.jpg", image);

  return 0;
}
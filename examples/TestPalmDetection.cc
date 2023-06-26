/*
 * @Author: chenjingyu
 * @Date: 2023-06-20 15:43:27
 * @LastEditTime: 2023-06-25 15:12:54
 * @Description: test palm detection
 * @FilePath: \Mediapipe-Hand\examples\TestPalmDetection.cc
 */
#include "TypeDefines.h"
#include "PalmDetector.h"
#include "LandmarkDetector.h"
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

  const char *palm_model_file = "../data/models/palm_detection.mnn";
  const char *landmark_model_file = "../data/models/hand_landmark.mnn";
  RotateType type = RotateType::CLOCKWISE_ROTATE_0;
  PalmDetector detector;
  LandmarkerDetector landmarker;
  if (!detector.LoadModel(palm_model_file) ||
      !landmarker.LoadModel(landmark_model_file)) {
    std::cout << "Failed load model." << std::endl;
    return -1;
  }
  detector.setSourceFormat(in.pixel_format);

  std::vector<ObjectInfo> objects;
  detector.Detect(in, type, objects);

  landmarker.setSourceFormat(in.pixel_format);
  landmarker.Detect(in, type, objects);
  int m = 0;
  for (const auto &object : objects) {
    cv::rectangle(image, cv::Point2f(object.tl.x, object.tl.y),
                  cv::Point2f(object.br.x, object.br.y),
                  cv::Scalar(255, 0, 255), 2);
    cv::putText(image, std::to_string(m),
                cv::Point2f(object.tl.x, object.tl.y), 1, 1.0,
                cv::Scalar(255, 0, 255));
    for (int i = 0; i < 21; ++i) {
      cv::Point pt = cv::Point(
        (int)object.landmarks[i].x,
        (int)object.landmarks[i].y
      );
      cv::circle(image, pt, 2, cv::Scalar(0, 255, 0));
    }
    m++;
  }


  cv::imshow("result", image);
  cv::waitKey(0);

  return 0;
}

/*
 * @Author: chenjingyu
 * @Date: 2023-06-20 15:43:27
 * @LastEditTime: 2023-08-01 18:02:13
 * @Description: test palm detection
 * @FilePath: \Mediapipe-MNN\examples\TestPalmDetection.cc
 */
#include "TypeDefines.h"
#include "PalmDetector.h"
#include "PalmLandmarkDetector.h"
#include <opencv2/opencv.hpp>
#include <cmath>

using namespace mirror;

int main(int argc, char *argv[]) {
  const char *image_file = "../data/images/hand.jpg";
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

  const char *palm_model_file = "../data/models/palm_detection_lite_fp16.mnn";
  const char *landmark_model_file = "../data/models/hand_landmark_lite_fp16.mnn";
  PalmDetector detector;
  PalmLandmarkDetector landmarker;
  if (!detector.LoadModel(palm_model_file) ||
      !landmarker.LoadModel(landmark_model_file)) {
    std::cout << "Failed load model." << std::endl;
    return -1;
  }
  detector.setFormat(in.pixel_format);

  std::vector<ObjectInfo> objects;
  detector.Detect(in, type, objects);

  landmarker.setFormat(in.pixel_format);
  landmarker.Detect(in, type, objects);
  for (const auto &object : objects) {
    cv::rectangle(image, cv::Point2f(object.rect.left, object.rect.top),
                  cv::Point2f(object.rect.right, object.rect.bottom),
                  cv::Scalar(255, 0, 255), 2);
    if (object.left_right == 1) {
      cv::putText(image, "left", cv::Point2f(object.rect.left, object.rect.top - 10), 1, 3.0,
                  cv::Scalar(255, 0, 255));
    } else {
      cv::putText(image, "right", cv::Point2f(object.rect.left, object.rect.top - 10), 1, 3.0,
                  cv::Scalar(255, 0, 255));
    }
    
    for (int i = 0; i < object.landmarks.size(); ++i) {
       cv::Point pt = cv::Point(
         (int)object.landmarks[i].x,                               
         (int)object.landmarks[i].y
      );
      cv::circle(image, pt, 2, cv::Scalar(255, 255, 0));
    }

    for (int i = 0; i < object.landmarks3d.size(); ++i) {
      cv::Point pt = cv::Point(
        (int)object.landmarks3d[i].x,
        (int)object.landmarks3d[i].y
      );
      cv::circle(image, pt, 2, cv::Scalar(0, 255, 0));
    }
  }


  cv::imshow("result", image);
  cv::waitKey(0);  
  cv::imwrite("../data/results/mediapipe_hand.jpg", image);

  return 0;
}

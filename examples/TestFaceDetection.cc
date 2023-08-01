/*
 * @Author: chenjingyu
 * @Date: 2023-07-29 17:05:18
 * @LastEditTime: 2023-08-01 18:02:47
 * @Description: Test face detection
 * @FilePath: \Mediapipe-MNN\examples\TestFaceDetection.cc
 */
#include "FaceDetector.h"
#include "FaceLandmarkDetector.h"
#include "TypeDefines.h"
#include <cmath>
#include <opencv2/opencv.hpp>


using namespace mirror;

int main(int argc, char *argv[]) {
  const char *image_file = "../data/images/face_tongue.jpg";
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

  const char *face_model_file =
      "../data/models/face_detection_short_range_fp16.mnn";
  const char *face_landmark_model_file =
      "../data/models/face_landmarks_detector_fp16.mnn";
  FaceDetector detector;
  FaceLandmarkDetector landmarker;
  if (!detector.LoadModel(face_model_file) ||
      !landmarker.LoadModel(face_landmark_model_file)) {
    std::cout << "Failed load model." << std::endl;
    return -1;
  }
  detector.setSourceFormat(in.pixel_format);
  landmarker.setSourceFormat(in.pixel_format);
  // detector.setUseFull();

  std::vector<ObjectInfo> objects;
  detector.Detect(in, type, objects);
  landmarker.Detect(in, type, objects);
  for (const auto &object : objects) {
    cv::rectangle(image, cv::Point2f(object.rect.left, object.rect.top),
                  cv::Point2f(object.rect.right, object.rect.bottom),
                  cv::Scalar(255, 0, 255), 2);
    for (int i = 0; i < object.landmarks.size(); ++i) {
      cv::Point pt = cv::Point((int)object.landmarks[i].x, (int)object.landmarks[i].y);
      cv::circle(image, pt, 2, cv::Scalar(255, 255, 0));
      cv::putText(image, std::to_string(i), pt, 1, 1.0, cv::Scalar(255, 0, 255));
    }
    for (int i = 0; i < object.landmarks3d.size(); ++i) {
      cv::Point pt = cv::Point((int)object.landmarks3d[i].x, (int)object.landmarks3d[i].y);
      cv::circle(image, pt, 2, cv::Scalar(255, 255, 0));
    }
    cv::putText(image, std::to_string(object.score),
                cv::Point2f(object.rect.left, object.rect.top + 20), 1, 1.0,
                cv::Scalar(0, 255, 255));
  }
  cv::imshow("result", image);
  cv::waitKey(0);

  return 0;
}
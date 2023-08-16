/*
 * @Author: chenjingyu
 * @Date: 2023-07-29 17:05:18
 * @LastEditTime: 2023-08-03 11:48:01
 * @Description: Test face detection
 * @FilePath: \Mediapipe-MNN\examples\TestFaceDetection.cc
 */
#include "FaceDetector.h"
#include "FaceLandmarkDetector.h"
#include "TypeDefines.h"
#include <cmath>
#include <opencv2/opencv.hpp>


using namespace mirror;
#define USE_VIDEO 

int main(int argc, char *argv[]) {
  const char *image_file = "../data/images/face_tongue.jpg";
  const char *face_model_file =  "../data/models/face_detection_short_range_fp16.mnn";
  const char *face_landmark_model_file = "../data/models/face_landmarks_detector_fp16.mnn";
  FaceDetector detector;
  FaceLandmarkDetector landmarker;
  if (!detector.LoadModel(face_model_file) ||
      !landmarker.LoadModel(face_landmark_model_file)) {
    std::cout << "Failed load model." << std::endl;
    return -1;
  }
  detector.setFormat(PixelFormat::BGR);
  landmarker.setFormat(PixelFormat::BGR);

  cv::Mat frame;
#ifdef USE_VIDEO
  cv::VideoWriter writer;
  cv::VideoCapture cam(0);
  if (!cam.isOpened()) {
    printf("Failed open camera.\n");
    return -2;
  }
  while (cv::waitKey(1) != 'q') {
    cam >> frame;
    if (!writer.isOpened())
      writer.open("mediapipe.mp4", CV_FOURCC('X', 'V', 'I', 'D'), 25,
                  cv::Size(frame.cols, frame.rows));
#else
    frame = cv::imread(image_file, 1);
#endif
    cv::Mat frame_src = frame.clone();
    if (frame.empty()) {
      printf("Empty frame.\n");
      return -3;
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


  std::vector<ObjectInfo> objects;
  detector.Detect(in, type, objects);
  landmarker.Detect(in, type, objects);
  for (const auto &object : objects) {
    cv::rectangle(frame_src, cv::Point2f(object.rect.left, object.rect.top),
                  cv::Point2f(object.rect.right, object.rect.bottom),
                  cv::Scalar(255, 0, 255), 2);
    for (int i = 0; i < object.landmarks.size(); ++i) {
      cv::Point pt = cv::Point((int)object.landmarks[i].x, (int)object.landmarks[i].y);
      cv::circle(frame_src, pt, 2, cv::Scalar(255, 255, 0));
      cv::putText(frame_src, std::to_string(i), pt, 1, 1.0, cv::Scalar(255, 0, 255));
    }
    for (int i = 0; i < object.landmarks3d.size(); ++i) {
      cv::Point pt = cv::Point((int)object.landmarks3d[i].x, (int)object.landmarks3d[i].y);
      cv::circle(frame_src, pt, 2, cv::Scalar(255, 255, 0));
    }
    cv::putText(frame_src, std::to_string(object.score),
                cv::Point2f(object.rect.left, object.rect.top + 20), 1, 1.0,
                cv::Scalar(0, 255, 255));
  }
#ifdef USE_VIDEO
    cv::imshow("frame", frame_src);
    int key = cv::waitKey(2);
    if (key == 27) break;
    writer << frame_src;
  }
  cam.release();
  writer.release();
#else
  cv::imshow("frame", frame_src);
  cv::waitKey(0);
#endif
  return 0;
}
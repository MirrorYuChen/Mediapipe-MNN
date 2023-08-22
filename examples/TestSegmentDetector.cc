#include "SegmentDetector.h"
#include "TypeDefines.h"
#include <opencv2/opencv.hpp>

using namespace mirror;
int main(int argc, char *argv[]) {
  const char *image_file = "../data/images/segment_demo.jpg";
  const char *model_file = "../data/models/deeplab_v3_fp16.mnn";

  std::unique_ptr<SegmentDetector> detector = nullptr;
  detector.reset(new SegmentDetector());

  if (!detector->LoadModel(model_file)) {
    std::cout << "failed load model: " << model_file << std::endl;
    return -1;
  }
  detector->setFormat(BGR);

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

  ImageHead mask;
  detector->Detect(in, type, mask);
  cv::Mat cv_mask = cv::Mat(mask.height, mask.width, CV_8UC3, mask.data).clone();
  cv::addWeighted(frame, 0.7f, cv_mask, 1.0f, 0, frame);

  cv::imshow("result", frame);
	cv::waitKey(0);

  cv::imwrite("../data/results/mediapipe_segment_result.jpg", frame);

  return 0;
}
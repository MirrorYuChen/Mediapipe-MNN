/*
 * @Author: chenjingyu
 * @Date: 2023-08-04 20:45:45
 * @LastEditTime: 2023-08-04 22:01:12
 * @Description: Test Image Embedding
 * @FilePath: \Mediapipe-MNN\examples\TestImageEmbedding.cc
 */
#include "ImageEmbedding.h"
#include "TypeDefines.h"
#include "Utils.h"
#include <cmath>
#include <opencv2/opencv.hpp>
using namespace mirror;

int main(int argc, char *argv[]) {
  const char *src_file = "../data/images/burger.jpg";
  const char *dst_file = "../data/images/burger_crop.jpg";
  cv::Mat image_src = cv::imread(src_file);
  cv::Mat image_dst = cv::imread(dst_file);
  if (image_src.empty() || image_dst.empty()) {
    std::cout << "failed load image." << std::endl;
    return -1;
  }
  RotateType type = RotateType::CLOCKWISE_ROTATE_0;
  if (type == CLOCKWISE_ROTATE_90) {
    cv::transpose(image_src, image_src);
    cv::transpose(image_dst, image_dst);
  } else if (type == CLOCKWISE_ROTATE_180) {
    cv::flip(image_src, image_src, -1);
    cv::flip(image_dst, image_dst, -1);
  }
  if (type == CLOCKWISE_ROTATE_270) {
    cv::transpose(image_src, image_src);
    cv::flip(image_src, image_src, 1);
    cv::transpose(image_dst, image_dst);
    cv::flip(image_dst, image_dst, 1);
  }

  ImageHead in_src, in_dst;
  in_src.data = image_src.data;
  in_src.height = image_src.rows;
  in_src.width = image_src.cols;
  in_src.width_step = image_src.step[0];
  in_src.pixel_format = PixelFormat::BGR;

  in_dst.data = image_dst.data;
  in_dst.height = image_dst.rows;
  in_dst.width = image_dst.cols;
  in_dst.width_step = image_dst.step[0];
  in_dst.pixel_format = PixelFormat::BGR;

  const char *embeded_model_file = "../data/models/embedder_small_fp16.mnn";
  ImageEmbedding detector;
  if (!detector.LoadModel(embeded_model_file)) {
    std::cout << "Failed load model." << std::endl;
    return -1;
  }
  detector.setSourceFormat(PixelFormat::BGR);

  Embedding embedding_src, embedding_dst;
  detector.Detect(in_src, type, embedding_src);
  detector.Detect(in_dst, type, embedding_dst);

  auto result = CosineSimilarity(embedding_src, embedding_dst);
  if (result.has_value()) {
    std::cout << "similarity: " << result.value() << std::endl;
  }



  return 0;
}
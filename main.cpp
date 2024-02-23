#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

// Generate histogram (cv::Mat -> 1d arr)
// Plot histogram (1d arr -> cv::Mat)
// Histogram peak (1d arr -> [int, int])
// Sliding window ()

int main()
{
    // Load image
    std::string image_path = "/root/adc/lane_line/step0.png";
    cv::Mat img_raw = cv::imread(image_path, cv::IMREAD_COLOR);

    if (img_raw.empty())
    {
        std::cout << "Could not read the image: " << image_path << std::endl;
        return 1;
    }

    // Preprocessing
    int img_width = 1200;
    int img_height = 900;
    cv::Mat img_resized, img_grayscale;

    cv::resize(img_raw, img_resized, cv::Size(img_width, img_height), cv::INTER_LINEAR);
    cv::cvtColor(img_resized, img_grayscale, cv::COLOR_BGR2GRAY);

    // Edge Detection
    cv::Mat img_blur, img_canny;

    cv::blur(img_grayscale, img_blur, cv::Size(5, 5));
    cv::Canny(img_blur, img_canny, 20, 100);

    // Mask
    cv::Mat mask = cv::Mat::zeros(img_canny.size(), img_canny.type());
    cv::Mat img_masked = cv::Mat::zeros(img_canny.size(), img_canny.type());

    int upper_bound = 650;
    int lower_bound = 830;
    int upper_width = 300;
    int lower_width = 900;

    std::vector<cv::Point2f> roi_points;
    roi_points.push_back(cv::Point2f((img_width - upper_width) / 2, upper_bound)); // Top left
    roi_points.push_back(cv::Point2f((img_width - lower_width) / 2, lower_bound)); // Bottom left
    roi_points.push_back(cv::Point2f((img_width + lower_width) / 2, lower_bound)); // Bottom right
    roi_points.push_back(cv::Point2f((img_width + upper_width) / 2, upper_bound)); // Top right

    std::vector<cv::Point> roi_display;
    cv::Mat(roi_points).convertTo(roi_display, CV_32F);
    cv::fillConvexPoly(mask, roi_display, cv::Scalar(255, 0, 0));
    img_canny.copyTo(img_masked, mask);

    // Get warp matrices
    std::vector<cv::Point2f> imagePoints;
    imagePoints.push_back(cv::Point2f(0, 0));                  // Top left
    imagePoints.push_back(cv::Point2f(0, img_height));         // Bottom left
    imagePoints.push_back(cv::Point2f(img_width, img_height)); // Bottom right
    imagePoints.push_back(cv::Point2f(img_width, 0));          // Top right

    cv::Mat warp_matrix = cv::getPerspectiveTransform(roi_points, imagePoints);
    cv::Mat unwarp_matrix = cv::getPerspectiveTransform(imagePoints, roi_points);

    // Warp image
    cv::Mat img_warped;
    cv::warpPerspective(img_canny, img_warped, warp_matrix, cv::Size(img_width, img_height));

    // Histogram
    cv::Mat histogram;
    const int *channel_numbers = {0};
    float channel_range[] = {0.0, 255.0};
    const float *channel_ranges = channel_range;
    int number_bins = 64;
    cv::calcHist(&img_warped, 1, channel_numbers, cv::Mat(), histogram, 1, &number_bins, &channel_ranges);

    // Display image
    //cv::imshow("Image", img_resized);
    //cv::imshow("hist", histogram);
    int k = cv::waitKey(0);
    std::cout << k << std::endl;
}

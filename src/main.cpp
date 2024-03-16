#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

#include "polyfit.h"
#include "binarize.h"
#include "generate.h"
#include "draw.h"
#include "birdeye.h"

// Image parameters
const int IMG_WIDTH = 960;
const int IMG_HEIGHT = 600;

// Define region of interest (ROI)
const int ROI_UPPER_BOUND = 450;
const int ROI_LOWER_BOUND = 550;
const int ROI_UPPER_WIDTH = 275;
const int ROI_LOWER_WIDTH = 600;

const std::vector<cv::Point2f> ROI_POINTS = {
    cv::Point2f((IMG_WIDTH - ROI_UPPER_WIDTH) / 2, ROI_UPPER_BOUND),
    cv::Point2f((IMG_WIDTH - ROI_LOWER_WIDTH) / 2, ROI_LOWER_BOUND),
    cv::Point2f((IMG_WIDTH + ROI_LOWER_WIDTH) / 2, ROI_LOWER_BOUND),
    cv::Point2f((IMG_WIDTH + ROI_UPPER_WIDTH) / 2, ROI_UPPER_BOUND),
};

// Define image warp region
const int WARP_OFFSET = 50;

const std::vector<cv::Point2f> WARP_POINTS = {
    cv::Point2f(WARP_OFFSET, 0),
    cv::Point2f(WARP_OFFSET, IMG_HEIGHT),
    cv::Point2f(IMG_WIDTH - WARP_OFFSET, IMG_HEIGHT),
    cv::Point2f(IMG_WIDTH - WARP_OFFSET, 0),
};

// Forward declarations
cv::Mat load_image(std::string img_path)
{
    // read image
    cv::Mat img = cv::imread(img_path, cv::IMREAD_COLOR);

    // check if image was read
    if (img.empty())
    {
        std::cerr << "Could not read the image: " << img_path << std::endl;
    }

    // resize image
    cv::Mat img_resized;
    cv::resize(img, img_resized, cv::Size(IMG_WIDTH, IMG_HEIGHT), cv::INTER_LINEAR);

    return img_resized;
}

cv::Mat detect_edges(cv::Mat img)
{
    cv::Mat img_gray;
    cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);

    cv::Mat img_blur;
    cv::blur(img_gray, img_blur, cv::Size(5, 5));

    cv::Mat img_canny;
    cv::Canny(img_blur, img_canny, 20, 100);

    return img_canny;
}

cv::Mat warp_image(cv::Mat img, std::vector<cv::Point2f> src_points, std::vector<cv::Point2f> dst_points)
{
    // get warp matrix
    cv::Mat matrix = cv::getPerspectiveTransform(src_points, dst_points);

    // warp image
    cv::Mat img_warped;
    cv::warpPerspective(img, img_warped, matrix, cv::Size(IMG_WIDTH, IMG_HEIGHT));

    return img_warped;
}

int get_hist_peak(cv::Mat img, int start, int end)
{
    int peak_idx = (end - start) / 2;
    float peak_val = 0;

    for (int col = start; col < end; col++)
    {
        float curr_val = 10 * (600 - cv::abs(600 - col)); // weight central columns more heavily

        // add each pixel value in the column
        for (int row = 0; row < IMG_HEIGHT; row++)
        {
            curr_val += img.at<uchar>(row, col);
        }

        // check if new peak
        if (curr_val > peak_val)
        {
            peak_idx = col;
            peak_val = curr_val;
        }
    }

    return peak_idx;
}

std::vector<cv::Point> slide_windows(cv::Mat img, int start, int count, int width)
{
    std::vector<cv::Point> points;

    for (int i = 0; i < count; i++)
    {
        int height = IMG_HEIGHT / count;

        // Calculate the window range based on method parameters
        int start_col = std::max(0, start - width / 2);
        int end_col = std::min(img.cols - 1, start + width / 2);

        // Create cropped window
        cv::Mat window = img(cv::Range(height * i, std::min(height * (i + 1), img.rows)), cv::Range(start_col, end_col));

        // Get mean point
        cv::Mat nonzero;
        findNonZero(window, nonzero);

        if (!nonzero.empty())
        {
            cv::Scalar mean = cv::mean(nonzero);
            points.push_back(cv::Point(mean[0] + start_col, mean[1] + height * i));
            start = mean[0] + start_col;
        }
    }

    // Add random points if none are found
    if (points.empty())
    {
        points.push_back(cv::Point(0, 0));
        points.push_back(cv::Point(10, 10));
        points.push_back(cv::Point(20, 20));
    }

    return points;
}

int main()
{
    // Load image
    cv::Mat img = load_image("/root/adc/lane_line/src/images/harder_challenge_video_frame_10.jpg");

    // Equalize
    cv::Mat ycrcb;

    cv::cvtColor(img, ycrcb, cv::COLOR_BGR2YCrCb);

    std::vector<cv::Mat> channels;
    cv::split(ycrcb, channels);

    cv::equalizeHist(channels[0], channels[0]);

    cv::Mat equalized;
    cv::merge(channels, ycrcb);

    cv::cvtColor(ycrcb, equalized, cv::COLOR_YCrCb2BGR);

    // Thresholding
    cv::Mat bin = binarize(equalized);

    // Warp image
    cv::Mat warped = warp_image(bin, ROI_POINTS, WARP_POINTS);

    // Get histogram peaks
    int left_peak = get_hist_peak(warped, 0, IMG_WIDTH / 2);
    int right_peak = get_hist_peak(warped, IMG_WIDTH / 2, IMG_WIDTH);

    // Get points with sliding windows
    std::vector<cv::Point> left_points = slide_windows(warped, left_peak, 20, 300);
    std::vector<cv::Point> right_points = slide_windows(warped, right_peak, 20, 300);

    // Get poly points
    std::vector<std::vector<cv::Point>> left_contours = {left_points};
    std::vector<std::vector<cv::Point>> right_contours = {right_points};
    double epsilon = 10.0;

    std::vector<cv::Point> approxPolyLeft;
    std::vector<cv::Point> approxPolyRight;

    cv::approxPolyDP(left_contours[0], approxPolyLeft, epsilon, false);
    cv::approxPolyDP(right_contours[0], approxPolyRight, epsilon, false);

    std::cout << "Left Points: " << approxPolyLeft << std::endl;
    std::cout << "Right Points: " << approxPolyRight << std::endl;

    std::vector<float> xs, ys;
    std::transform(approxPolyLeft.begin(), approxPolyLeft.end(), std::back_inserter(xs), [](const auto &p)
                   { return p.x; });
    std::transform(approxPolyLeft.begin(), approxPolyLeft.end(), std::back_inserter(ys), [](const auto &p)
                   { return p.y; });
    std::vector<float> coeffs = polyfit_boost(ys, xs, 2);
    Coefficient left_coeff = Coefficient(coeffs[2], coeffs[1], coeffs[0]);

    for (int i = 0; i < coeffs.size(); i++)
    {
        std::cout << "coeffs: " << coeffs[i] << std::endl;
    }

    // Polylines
    cv::Mat bev_img = warped.clone();
    std::vector<cv::Point> lpoints = generate_line_points(bev_img.size(), left_coeff);

    cv::Mat unwarp_m = cv::getPerspectiveTransform(WARP_POINTS, ROI_POINTS);

    // BEVWarper bev { bev_img.size() };
    std::vector<cv::Point2f> points_f;
    std::copy(lpoints.begin(), lpoints.end(), std::back_inserter(points_f));

    std::vector<cv::Point2f> unwarped_points;
    cv::perspectiveTransform(points_f, unwarped_points, unwarp_m);

    std::vector<cv::Point2i> unwarped_points_i;
    std::copy(unwarped_points.begin(), unwarped_points.end(), std::back_inserter(unwarped_points_i));

    // Unwarp image
    cv::Mat unwarped = warp_image(bev_img, WARP_POINTS, ROI_POINTS);
    draw_points(unwarped, unwarped_points_i, cv::Scalar(0, 0, 255));

    // Combine polylines with original image
    cv::Mat result;
    cv::addWeighted(img, 0.5, unwarped, 0.5, 0.0, result);

    // Display image
    cv::imshow("Window 2", bev_img);
    cv::imshow("Window 3", bin);
    cv::imshow("Window 4", result);
    int key = cv::waitKey(0);
    std::cout << key << std::endl;
}

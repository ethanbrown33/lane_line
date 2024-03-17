#include <iostream>
#include <iomanip>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

#include "read.h"
#include "binarize.h"
#include "birdeye.h"
#include "detect.h"
#include "translate.h"
#include "draw.h"
#include "coefficient.h"
#include "polyfit.h"
#include "smoothen.h"
#include "generate.h"
#include "offset.h"
#include "radius.h"
#include "display.h"

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

std::vector<cv::Point2i> slide_windows(cv::Mat img, int start, int count, int width)
{
    std::vector<cv::Point2i> points;

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
            // points.push_back(cv::Point(mean[0] + start_col, mean[1] + height * i));
            points.push_back(cv::Point(mean[0] + start_col, 30 + height * i));
            start = mean[0] + start_col;
        }
    }

    // Add random points if none are found
    if (points.empty())
    {
        points.push_back(cv::Point2i(0, 0));
        points.push_back(cv::Point2i(10, 10));
        points.push_back(cv::Point2i(20, 20));
    }

    return points;
}

enum Mode : uint8_t {
    IMAGE, VIDEO
};

int main()
{
    // Load image
    cv::Mat img = load_image("/root/adc/lane_line/src/images/highway.png");

    // Threshold
    // img.convertTo(img, -1, 2, 50);
    const auto bin_img = binarize(img);

    // Warp image to BEV
    BEVWarper bev{bin_img.size()};
    const auto bev_img = bev.warp(bin_img);

    // Detect lane points on BEV
    std::vector<cv::Point2i> lpoints_bev, rpoints_bev;
    {
        // Get histogram peaks
        int left_peak = get_hist_peak(bev_img, 0, IMG_WIDTH / 2);
        int right_peak = get_hist_peak(bev_img, IMG_WIDTH / 2, IMG_WIDTH);

        // Get points with sliding windows
        lpoints_bev = slide_windows(bev_img, left_peak, 10, 100);
        rpoints_bev = slide_windows(bev_img, right_peak, 10, 100);
    }

    // Fit points with second polynomial order
    Coefficient left_coeff, right_coeff;
    {
        std::vector<float> xs, ys;
        std::transform(lpoints_bev.begin(), lpoints_bev.end(), std::back_inserter(xs), [](const auto &p)
                       { return p.x; });
        std::transform(lpoints_bev.begin(), lpoints_bev.end(), std::back_inserter(ys), [](const auto &p)
                       { return p.y; });
        std::vector<float> coeffs = polyfit_boost(ys, xs, 2);
        left_coeff = Coefficient(coeffs[2], coeffs[1], coeffs[0]);
    }
    {
        std::vector<float> xs, ys;
        std::transform(rpoints_bev.begin(), rpoints_bev.end(), std::back_inserter(xs), [](const auto &p)
                       { return p.x; });
        std::transform(rpoints_bev.begin(), rpoints_bev.end(), std::back_inserter(ys), [](const auto &p)
                       { return p.y; });
        std::vector<float> coeffs = polyfit_boost(ys, xs, 2);
        right_coeff = Coefficient(coeffs[2], coeffs[1], coeffs[0]);
    }

    // Generate line points on BEV
    lpoints_bev = generate_line_points(bev_img.size(), left_coeff);
    rpoints_bev = generate_line_points(bev_img.size(), right_coeff);

    // Unwarp bev points to original image
    std::vector<cv::Point2i> lpoints = bev.unwarp_points(lpoints_bev);
    std::vector<cv::Point2i> rpoints = bev.unwarp_points(rpoints_bev);

    // Calculate curve radius and vehicle offset from the lane center
    float offset, radius;
    {
        int ego_x = bin_img.cols / 2;
        int bottom_mid_x = (lpoints[0].x + rpoints[0].x) / 2;
        offset = calculate_offset_in_meter(ego_x, bottom_mid_x);
        float radius_left = calculate_curve_radius(left_coeff);
        float radius_right = calculate_curve_radius(right_coeff);
        radius = (radius_left + radius_right) / 2;
    }

    // Visualize lane
    auto vis_image = img.clone();
    std::vector<cv::Point2i> lane_points(lpoints.size());
    std::reverse_copy(lpoints.begin(), lpoints.end(), lane_points.begin());
    lane_points.insert(lane_points.end(), rpoints.begin(), rpoints.end());
    draw_polygon(vis_image, lane_points);

    // Draw lane boundaries
    draw_curve(vis_image, lpoints, cv::Scalar(0, 0, 255));
    draw_curve(vis_image, rpoints, cv::Scalar(255, 0, 0));

    // Visualize binary image
    {
        cv::Mat small_img;
        cv::resize(bin_img, small_img, bin_img.size() / 5);
        overlay(small_img, vis_image, 10, 10);
    }

    // Visualize bev image
    {
        cv::Mat small_img;
        cv::resize(bev_img, small_img, bev_img.size() / 5);
        overlay(small_img, vis_image, 10 * 2 + small_img.cols, 10);
    }

    // Visualize bev image with detected points
    {
        cv::Mat bev_img_color = bev_img.clone();
        draw_points(bev_img_color, lpoints_bev, cv::Scalar(0, 0, 255));
        draw_points(bev_img_color, rpoints_bev, cv::Scalar(255, 0, 0));
        cv::Mat small_img;
        cv::resize(bev_img_color, small_img, bev_img_color.size() / 5);
        overlay(small_img, vis_image, 10 * 3 + small_img.cols * 2, 10);
    }

    // Visualize offset
    {
        std::ostringstream stream;
        stream << "Center offset = " << std::fixed << std::setprecision(3) << offset << " m";
        std::string text = stream.str();
        cv::putText(vis_image, text, cv::Point2i(20, 150),
                    cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8,
                    cv::Scalar(255, 0, 0));
    }

    // Visualize curve radius
    {
        std::ostringstream stream;
        stream << "Curve radius = " << std::fixed << std::setprecision(1) << radius << " m";
        std::string text = stream.str();
        cv::putText(vis_image, text, cv::Point2i(20, 170),
                    cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8,
                    cv::Scalar(255, 0, 0));
    }

    // Display image
    cv::imshow("Window 1", vis_image);

    int key = cv::waitKey(0);
    std::cout << key << std::endl;
}

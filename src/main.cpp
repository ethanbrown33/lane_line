#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

std::vector<cv::Point> slide_windows(cv::Mat img, int x, int box_w, int box_h);

// Function to draw a curve using polylines
std::vector<cv::Point> draw_curve(cv::Mat image, std::vector<cv::Point> points)
{
    // Extract x and y values from points
    std::vector<double> x, y;
    for (cv::Point point : points) {
        x.push_back(point.x);
        y.push_back(point.y);
    }

    // Create the matrices for the least squares method
    cv::Mat A = cv::Mat(points.size(), 3, CV_64F);
    cv::Mat B = cv::Mat(points.size(), 1, CV_64F);

    for (int i = 0; i < points.size(); ++i) {
        A.at<double>(i, 0) = x[i] * x[i];
        A.at<double>(i, 1) = x[i];
        A.at<double>(i, 2) = 1.0;
        B.at<double>(i, 0) = y[i];
    }

    // Solve the least squares problem
    cv::Mat solution;
    cv::solve(A, B, solution, cv::DECOMP_QR);

    // Extract the coefficients
    double a = solution.at<double>(0, 0);
    double b = solution.at<double>(1, 0);
    double c = solution.at<double>(2, 0);

    // Generate points for the parabolic line
    std::vector<cv::Point> parabolicPoints;
    for (int col = 0; col < image.cols; ++col) {
        int row = static_cast<int>(a * col * col + b * col + c);
        if (row >= 0 && row < image.rows) {
            parabolicPoints.emplace_back(col, row);
        }
    }

    // Draw the parabolic line on the image using polylines
    if (!parabolicPoints.empty()) {
        const cv::Point* data = &parabolicPoints[0];
        int size = static_cast<int>(parabolicPoints.size());
        polylines(image, &data, &size, 1, false, cv::Scalar(0, 255, 0), 10);
    }

    return parabolicPoints;
}


int main()
{
    // Load image
    // std::string image_path = "/root/adc/lane_line/src/images/step0.png";
    // std::string image_path = "/root/adc/lane_line/src/images/challenge_video_frame_1.jpg";
    // std::string image_path = "/root/adc/lane_line/src/images/harder_challenge_video_frame_10.jpg";
    std::string image_path = "/root/adc/lane_line/src/images/frame0.jpg";
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
    int offset = 200;
    std::vector<cv::Point2f> imagePoints;
    imagePoints.push_back(cv::Point2f(offset, 0));                  // Top left
    imagePoints.push_back(cv::Point2f(offset, img_height));         // Bottom left
    imagePoints.push_back(cv::Point2f(img_width - offset, img_height)); // Bottom right
    imagePoints.push_back(cv::Point2f(img_width - offset, 0));          // Top right

    cv::Mat warp_matrix = cv::getPerspectiveTransform(roi_points, imagePoints);
    cv::Mat unwarp_matrix = cv::getPerspectiveTransform(imagePoints, roi_points);

    // Warp image
    cv::Mat img_warped;
    cv::warpPerspective(img_canny, img_warped, warp_matrix, cv::Size(img_width, img_height));

    // Get histogram vector
    std::vector<float> hist_vec(img_width, 0);
    for (int i = 0; i < img_width; i++)
    {
        for (int j = 0; j < img_height; j++)
        {
            cv::Point coord = cv::Point(j, i);
            hist_vec[i] += img_warped.at<uchar>(j, i);
        }

        // priority adjustment (add more value to center columns)
        hist_vec[i] += 10 * (600 - cv::abs(600 - i));
    }

    // Calculate histogram peaks
    int left_peak = img_width / 4;
    int right_peak = 3 * img_width / 4;

    for (int i = 0; i < img_width / 2; i++)
    {
        if (hist_vec[i] > hist_vec[left_peak])
        {
            left_peak = i;
        }
    }

    for (int i = img_width / 2; i < img_width; i++)
    {
        if (hist_vec[i] > hist_vec[right_peak])
        {
            right_peak = i;
        }
    }

    std::cout << "Left peak: " << left_peak << std::endl;
    std::cout << "Right peak: " << right_peak << std::endl;

    // histogram image
    cv::Mat hist_img = cv::Mat::zeros(img_warped.size(), CV_8U);

    cv::line(hist_img, cv::Point(left_peak, 0), cv::Point(left_peak, img_height), cv::Scalar(255, 0, 0), 10);
    cv::line(hist_img, cv::Point(right_peak, 0), cv::Point(right_peak, img_height), cv::Scalar(255, 0, 0), 10);

    for (int i = 0; i < img_width; i++)
    {
        float line_height = img_height - 4 * img_height * hist_vec[i] / (255 * img_height);
        cv::line(hist_img, cv::Point(i, img_height), cv::Point(i, line_height), cv::Scalar(255, 0, 0), 1);
    }

    // sliding windows
    std::vector<cv::Point> windows_left = slide_windows(img_warped, left_peak, 300, 100);
    std::vector<cv::Point> windows_right = slide_windows(img_warped, right_peak, 300, 100);

    for (int i = 0; i < windows_left.size(); i++)
    {
        std::cout << "Left point " << i << ": " << windows_left[i].x << "," << windows_left[i].y << std::endl;
        cv::circle(img_warped, windows_left[i], 30, cv::Scalar(255, 0, 0), -1);
    }

    for (int i = 0; i < windows_right.size(); i++)
    {
        std::cout << "Right point " << i << ": " << windows_right[i].x << "," << windows_right[i].y << std::endl;
        cv::circle(img_warped, windows_right[i], 30, cv::Scalar(255, 0, 0), -1);
    }

    cv::imshow("Image1", img_warped);

    // plot curves
    cv::Mat curveplot = cv::Mat::zeros(img_warped.size(), CV_8UC3);

    std::vector<cv::Point> left_plot = draw_curve(curveplot, windows_left);
    std::vector<cv::Point> right_plot = draw_curve(curveplot, windows_right);
    cv::imshow("Image2", curveplot);

    // unwarp image
    cv::Mat img_unwarped;
    cv::warpPerspective(curveplot, img_unwarped, unwarp_matrix, cv::Size(1200, 900));
    cv::imshow("Image3", img_unwarped);

    // combine images
    cv::Mat img_combined;
    cv::add(img_unwarped, img_resized, img_combined);

    cv::imshow("Image4", img_combined);


    int k = cv::waitKey(0);
    std::cout << k << std::endl;
}

std::vector<cv::Point> slide_windows(cv::Mat img, int x, int box_w, int box_h)
{
    std::vector<cv::Point> points;

    for (int i = 0; i < 900 / box_h; i++)
    {
        // Calculate the window range based on method parameters
        int start_col = std::max(0, x - box_w / 2);
        int end_col = std::min(img.cols - 1, x + box_w / 2);

        cv::Mat window = img(cv::Range(box_h * i, std::min(box_h * (i + 1), img.rows)),
                             cv::Range(start_col, end_col));

        // Get mean point
        cv::Mat nonzero;
        findNonZero(window, nonzero);

        if (!nonzero.empty())
        {
            cv::Scalar mean = cv::mean(nonzero);
            points.push_back(cv::Point(mean[0] + start_col, mean[1] + box_h * i));
        }
    }

    if (points.empty())
    {
        points.push_back(cv::Point(0, 0));
        points.push_back(cv::Point(10, 10));
        points.push_back(cv::Point(20, 20));
    }

    return points;
}


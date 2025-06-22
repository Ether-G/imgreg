#include <opencv2/opencv.hpp>
#include <iostream>
#include <random>

int main(int argc, char** argv) {
    if (argc < 5) {
        std::cout << "Usage: " << argv[0] << " <width> <height> <dx> <dy> [pattern_type] [complexity]\n";
        std::cout << "Pattern types: 0=box, 1=circle, 2=text, 3=gradient, 4=noise, 5=complex\n";
        std::cout << "Complexity: 0=simple, 1=medium, 2=high\n";
        return 1;
    }
    
    int width = std::stoi(argv[1]);
    int height = std::stoi(argv[2]);
    int dx = std::stoi(argv[3]);
    int dy = std::stoi(argv[4]);
    int pattern_type = (argc > 5) ? std::stoi(argv[5]) : 0;
    int complexity = (argc > 6) ? std::stoi(argv[6]) : 0;

    std::cout << "Generating " << width << "x" << height << " images with shift: (" << dx << ", " << dy << ")\n";
    std::cout << "Pattern type: " << pattern_type << ", Complexity: " << complexity << std::endl;

    // Create base image
    cv::Mat base = cv::Mat::zeros(height, width, CV_8UC1);
    
    // Generate different patterns
    switch (pattern_type) {
        case 0: // Box pattern
            cv::rectangle(base, cv::Point(width/4, height/4), cv::Point(3*width/4, 3*height/4), cv::Scalar(255), -1);
            if (complexity > 0) {
                cv::rectangle(base, cv::Point(width/3, height/3), cv::Point(2*width/3, 2*height/3), cv::Scalar(0), -1);
            }
            if (complexity > 1) {
                cv::circle(base, cv::Point(width/2, height/2), width/8, cv::Scalar(255), -1);
            }
            break;
            
        case 1: // Circle pattern
            cv::circle(base, cv::Point(width/2, height/2), width/3, cv::Scalar(255), -1);
            if (complexity > 0) {
                cv::circle(base, cv::Point(width/2, height/2), width/6, cv::Scalar(0), -1);
            }
            if (complexity > 1) {
                cv::circle(base, cv::Point(width/2, height/2), width/12, cv::Scalar(255), -1);
            }
            break;
            
        case 2: // Text pattern
            cv::putText(base, "TEST", cv::Point(width/4, height/2), cv::FONT_HERSHEY_SIMPLEX, 
                       std::max(1.0, width/100.0), cv::Scalar(255), 2);
            if (complexity > 0) {
                cv::putText(base, "123", cv::Point(width/2, height/2), cv::FONT_HERSHEY_SIMPLEX, 
                           std::max(0.5, width/200.0), cv::Scalar(255), 1);
            }
            break;
            
        case 3: // Gradient pattern
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    int value = (x * 255) / width;
                    if (complexity > 0) {
                        value = (value + (y * 255) / height) / 2;
                    }
                    if (complexity > 1) {
                        value = value + 50 * sin(x * 0.1) * cos(y * 0.1);
                    }
                    base.at<uchar>(y, x) = cv::saturate_cast<uchar>(value);
                }
            }
            break;
            
        case 4: // Noise pattern
            {
                std::random_device rd;
                std::mt19937 gen(rd());
                std::uniform_int_distribution<> dis(0, 255);
                
                for (int y = 0; y < height; y++) {
                    for (int x = 0; x < width; x++) {
                        int noise = dis(gen);
                        if (complexity > 0) {
                            // Add some structure
                            if ((x + y) % 20 < 10) noise = 255 - noise;
                        }
                        if (complexity > 1) {
                            // Add patterns
                            if (x % 30 < 15 && y % 30 < 15) noise = 255;
                        }
                        base.at<uchar>(y, x) = noise;
                    }
                }
            }
            break;
            
        case 5: // Complex pattern (combination)
            // Main rectangle
            cv::rectangle(base, cv::Point(width/4, height/4), cv::Point(3*width/4, 3*height/4), cv::Scalar(255), -1);
            // Inner circle
            cv::circle(base, cv::Point(width/2, height/2), width/6, cv::Scalar(0), -1);
            // Text
            cv::putText(base, "IMG", cv::Point(width/3, height/2), cv::FONT_HERSHEY_SIMPLEX, 
                       std::max(0.8, width/150.0), cv::Scalar(255), 2);
            // Add some noise
            if (complexity > 0) {
                std::random_device rd;
                std::mt19937 gen(rd());
                std::normal_distribution<> dis(0, 20);
                for (int y = 0; y < height; y++) {
                    for (int x = 0; x < width; x++) {
                        int noise = cv::saturate_cast<int>(base.at<uchar>(y, x) + dis(gen));
                        base.at<uchar>(y, x) = cv::saturate_cast<uchar>(noise);
                    }
                }
            }
            break;
    }

    // Create shifted image
    cv::Mat shifted = cv::Mat::zeros(height, width, CV_8UC1);
    
    // Calculate the region that will be copied
    int src_x1 = std::max(0, dx);
    int src_y1 = std::max(0, dy);
    int src_x2 = std::min(width, width + dx);
    int src_y2 = std::min(height, height + dy);
    
    int dst_x1 = std::max(0, -dx);
    int dst_y1 = std::max(0, -dy);
    int dst_x2 = std::min(width, width - dx);
    int dst_y2 = std::min(height, height - dy);
    
    // Copy the overlapping region
    cv::Rect src_roi(src_x1, src_y1, src_x2 - src_x1, src_y2 - src_y1);
    cv::Rect dst_roi(dst_x1, dst_y1, dst_x2 - dst_x1, dst_y2 - dst_y1);
    
    if (src_roi.width > 0 && src_roi.height > 0 && dst_roi.width > 0 && dst_roi.height > 0) {
        base(src_roi).copyTo(shifted(dst_roi));
    }

    // Save images
    cv::imwrite("source.png", base);
    cv::imwrite("shifted.png", shifted);

    std::cout << "Images saved as source.png and shifted.png\n";
    std::cout << "Source ROI: (" << src_roi.x << ", " << src_roi.y << ", " << src_roi.width << ", " << src_roi.height << ")\n";
    std::cout << "Dest ROI: (" << dst_roi.x << ", " << dst_roi.y << ", " << dst_roi.width << ", " << dst_roi.height << ")\n";
    
    return 0;
} 
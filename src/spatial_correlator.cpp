#include "spatial_correlator.h"
#include "thread_manager.h"
#include "memory_pool.h"
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <cmath>

// SIMD includes with fallbacks
#ifdef __AVX512F__
#include <immintrin.h>
#define USE_AVX512
#elif defined(__AVX2__)
#include <immintrin.h>
#define USE_AVX2
#elif defined(__AVX__)
#include <immintrin.h>
#define USE_AVX
#endif

namespace imgreg {

SpatialCorrelator::SpatialCorrelator()
    : thread_count_(1),
      window_width_(64),
      window_height_(64),
      step_x_(1),
      step_y_(1),
      simd_enabled_(true),
      last_execution_time_(std::chrono::microseconds(0)),
      last_memory_usage_(0) {
    
    thread_manager_ = std::make_shared<ThreadManager>(thread_count_);
}

SpatialCorrelator::~SpatialCorrelator() = default;

CorrelationResult SpatialCorrelator::correlate(const cv::Mat& image_a, const cv::Mat& image_b) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Convert images to CV_32F for processing
    cv::Mat img_a_float, img_b_float;
    image_a.convertTo(img_a_float, CV_32F);
    image_b.convertTo(img_b_float, CV_32F);
    
    // Simple sliding window correlation to find shift
    int max_shift = std::min(img_a_float.cols, img_a_float.rows) / 4; // Limit search range
    double best_correlation = -1.0;
    int best_dx = 0, best_dy = 0;
    
    // Search for the best correlation
    for (int dy = -max_shift; dy <= max_shift; dy++) {
        for (int dx = -max_shift; dx <= max_shift; dx++) {
            double correlation = 0.0;
            int valid_pixels = 0;
            
            // Compute correlation for this shift
            for (int y = 0; y < img_a_float.rows; y++) {
                for (int x = 0; x < img_a_float.cols; x++) {
                    int x2 = x + dx;
                    int y2 = y + dy;
                    
                    if (x2 >= 0 && x2 < img_a_float.cols && y2 >= 0 && y2 < img_a_float.rows) {
                        correlation += img_a_float.at<float>(y, x) * img_b_float.at<float>(y2, x2);
                        valid_pixels++;
                    }
                }
            }
            
            if (valid_pixels > 0) {
                correlation /= valid_pixels;
                if (correlation > best_correlation) {
                    best_correlation = correlation;
                    best_dx = dx;
                    best_dy = dy;
                }
            }
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    last_execution_time_ = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    // Create result
    CorrelationResult result;
    result.method_name = "spatial";
    result.correlation_value = best_correlation;
    result.offset_x = -best_dx;
    result.offset_y = -best_dy;
    result.execution_time = last_execution_time_;
    result.memory_usage_bytes = img_a_float.total() * img_a_float.elemSize() * 2;
    result.thread_count = thread_count_;
    
    // Create a simple correlation map
    std::vector<std::vector<double>> correlation_map(1, std::vector<double>(1, best_correlation));
    result.correlation_map = correlation_map;
    
    return result;
}

double SpatialCorrelator::computeNormalizedCrossCorrelation(const cv::Mat& template_img, 
                                                           const cv::Mat& search_img,
                                                           int offset_x, int offset_y) {
    // Extract search window
    cv::Mat search_window = extractWindow(search_img, offset_x, offset_y, 
                                         template_img.cols, template_img.rows);
    
    // Normalize search window
    normalizeWindow(search_window);
    
    // Compute means
    cv::Scalar template_mean = cv::mean(template_img);
    cv::Scalar search_mean = cv::mean(search_window);
    
    // Compute correlation
    double numerator = 0.0;
    double template_var = 0.0;
    double search_var = 0.0;
    
    for (int y = 0; y < template_img.rows; ++y) {
        for (int x = 0; x < template_img.cols; ++x) {
            double template_val = template_img.at<double>(y, x) - template_mean[0];
            double search_val = search_window.at<double>(y, x) - search_mean[0];
            
            numerator += template_val * search_val;
            template_var += template_val * template_val;
            search_var += search_val * search_val;
        }
    }
    
    double denominator = std::sqrt(template_var * search_var);
    
    if (denominator < 1e-10) {
        return 0.0;
    }
    
    return numerator / denominator;
}

double SpatialCorrelator::computeCorrelationSIMD(const cv::Mat& template_img, 
                                                const cv::Mat& search_img,
                                                int offset_x, int offset_y) {
    // Extract search window
    cv::Mat search_window = extractWindow(search_img, offset_x, offset_y, 
                                         template_img.cols, template_img.rows);
    
    // Normalize search window
    normalizeWindow(search_window);
    
    // Compute means
    cv::Scalar template_mean = cv::mean(template_img);
    cv::Scalar search_mean = cv::mean(search_window);
    
    // SIMD-optimized correlation computation
    double numerator = 0.0;
    double template_var = 0.0;
    double search_var = 0.0;
    
    int total_pixels = template_img.rows * template_img.cols;
    const double* template_data = template_img.ptr<double>();
    const double* search_data = search_window.ptr<double>();
    
#ifdef USE_AVX512
    int simd_pixels = total_pixels - (total_pixels % 8); // Process 8 doubles at a time
    
    // SIMD computation
    // SIMD computation for the main part
    for (int i = 0; i < simd_pixels; i += 8) {
        __m512d template_vec = _mm512_loadu_pd(&template_data[i]);
        __m512d search_vec = _mm512_loadu_pd(&search_data[i]);
        
        __m512d template_mean_vec = _mm512_set1_pd(template_mean[0]);
        __m512d search_mean_vec = _mm512_set1_pd(search_mean[0]);
        
        __m512d template_diff = _mm512_sub_pd(template_vec, template_mean_vec);
        __m512d search_diff = _mm512_sub_pd(search_vec, search_mean_vec);
        
        __m512d product = _mm512_mul_pd(template_diff, search_diff);
        __m512d template_sq = _mm512_mul_pd(template_diff, template_diff);
        __m512d search_sq = _mm512_mul_pd(search_diff, search_diff);
        
        // Horizontal sum
        numerator += _mm512_reduce_add_pd(product);
        template_var += _mm512_reduce_add_pd(template_sq);
        search_var += _mm512_reduce_add_pd(search_sq);
    }
#elif defined(USE_AVX2) || defined(USE_AVX)
    int simd_pixels = total_pixels - (total_pixels % 4); // Process 4 doubles at a time
    
    // SIMD computation for the main part
    for (int i = 0; i < simd_pixels; i += 4) {
        __m256d template_vec = _mm256_loadu_pd(&template_data[i]);
        __m256d search_vec = _mm256_loadu_pd(&search_data[i]);
        
        __m256d template_mean_vec = _mm256_set1_pd(template_mean[0]);
        __m256d search_mean_vec = _mm256_set1_pd(search_mean[0]);
        
        __m256d template_diff = _mm256_sub_pd(template_vec, template_mean_vec);
        __m256d search_diff = _mm256_sub_pd(search_vec, search_mean_vec);
        
        __m256d product = _mm256_mul_pd(template_diff, search_diff);
        __m256d template_sq = _mm256_mul_pd(template_diff, template_diff);
        __m256d search_sq = _mm256_mul_pd(search_diff, search_diff);
        
        // Horizontal sum
        double product_arr[4], template_sq_arr[4], search_sq_arr[4];
        _mm256_storeu_pd(product_arr, product);
        _mm256_storeu_pd(template_sq_arr, template_sq);
        _mm256_storeu_pd(search_sq_arr, search_sq);
        
        for (int j = 0; j < 4; ++j) {
            numerator += product_arr[j];
            template_var += template_sq_arr[j];
            search_var += search_sq_arr[j];
        }
    }
#else
    // Fallback to scalar computation
    int simd_pixels = 0;
#endif
    
    // Handle remaining pixels
    for (int i = simd_pixels; i < total_pixels; ++i) {
        double template_val = template_data[i] - template_mean[0];
        double search_val = search_data[i] - search_mean[0];
        
        numerator += template_val * search_val;
        template_var += template_val * template_val;
        search_var += search_val * search_val;
    }
    
    double denominator = std::sqrt(template_var * search_var);
    
    if (denominator < 1e-10) {
        return 0.0;
    }
    
    return numerator / denominator;
}

std::pair<int, int> SpatialCorrelator::findPeakLocation(const std::vector<std::vector<double>>& correlation_map) {
    double max_val = correlation_map[0][0];
    int max_x = 0, max_y = 0;
    
    for (size_t y = 0; y < correlation_map.size(); ++y) {
        for (size_t x = 0; x < correlation_map[y].size(); ++x) {
            if (correlation_map[y][x] > max_val) {
                max_val = correlation_map[y][x];
                max_x = static_cast<int>(x);
                max_y = static_cast<int>(y);
            }
        }
    }
    
    return {max_x, max_y};
}

SpatialCorrelator::CorrelationWindow SpatialCorrelator::findPeakBranchless(const std::vector<std::vector<double>>& correlation_map) {
    CorrelationWindow peak;
    
    for (size_t y = 0; y < correlation_map.size(); ++y) {
        for (size_t x = 0; x < correlation_map[y].size(); ++x) {
            // Branch-free maximum finding using arithmetic operations
            double current_val = correlation_map[y][x];
            int mask = (current_val > peak.correlation_value) ? 1 : 0;
            
            peak.correlation_value = mask * current_val + (1 - mask) * peak.correlation_value;
            peak.x = mask * static_cast<int>(x) + (1 - mask) * peak.x;
            peak.y = mask * static_cast<int>(y) + (1 - mask) * peak.y;
        }
    }
    
    return peak;
}

cv::Mat SpatialCorrelator::extractWindow(const cv::Mat& image, int x, int y, int width, int height) {
    cv::Rect roi(x, y, width, height);
    return image(roi).clone();
}

void SpatialCorrelator::normalizeWindow(cv::Mat& window) {
    cv::Scalar mean = cv::mean(window);
    cv::Scalar stddev;
    cv::meanStdDev(window, cv::noArray(), stddev);
    
    if (stddev[0] > 1e-10) {
        window = (window - mean[0]) / stddev[0];
    }
}

void SpatialCorrelator::setThreadCount(int thread_count) {
    thread_count_ = thread_count;
    if (thread_manager_) {
        thread_manager_->setThreadCount(thread_count);
    }
}

void SpatialCorrelator::setWindowSize(int width, int height) {
    window_width_ = width;
    window_height_ = height;
}

void SpatialCorrelator::setStepSize(int step_x, int step_y) {
    step_x_ = step_x;
    step_y_ = step_y;
}

void SpatialCorrelator::enableSIMD(bool enable) {
    simd_enabled_ = enable;
}

void SpatialCorrelator::setMemoryPool(std::shared_ptr<MemoryPool> pool) {

}

} // namespace imgreg 
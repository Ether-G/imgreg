#include "pearson_correlator.h"
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

PearsonCorrelator::PearsonCorrelator()
    : thread_count_(1),
      normalization_type_("zscore"),
      simd_enabled_(true),
      last_execution_time_(std::chrono::microseconds(0)),
      last_memory_usage_(0) {
    
    thread_manager_ = std::make_shared<ThreadManager>(thread_count_);
    temp_buffer_.reserve(1024 * 1024); // Reserve 1MB
}

PearsonCorrelator::~PearsonCorrelator() = default;

CorrelationResult PearsonCorrelator::correlate(const cv::Mat& image_a, const cv::Mat& image_b) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Validate image dimensions
    if (!validateImageDimensions(image_a, image_b)) {
        throw std::invalid_argument("Images must have the same dimensions for Pearson correlation");
    }
    
    // Normalize images
    cv::Mat normalized_a = normalizeImage(image_a);
    cv::Mat normalized_b = normalizeImage(image_b);
    
    // Compute Pearson correlation coefficient
    double correlation_coefficient;
    if (simd_enabled_) {
        correlation_coefficient = computeCorrelationSIMD(normalized_a, normalized_b);
    } else {
        correlation_coefficient = computePearsonCoefficient(normalized_a, normalized_b);
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    last_execution_time_ = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    // Create result
    CorrelationResult result;
    result.method_name = "pearson";
    result.correlation_value = std::abs(correlation_coefficient);
    result.offset_x = 0; // Pearson correlation does not provide spatial offset
    result.offset_y = 0;
    result.execution_time = last_execution_time_;
    result.memory_usage_bytes = image_a.rows * image_a.cols * sizeof(double) * 2; // Two normalized images
    result.thread_count = thread_count_;
    result.pearson_coefficient = correlation_coefficient;
    
    return result;
}

double PearsonCorrelator::computePearsonCoefficient(const cv::Mat& image_a, const cv::Mat& image_b) {
    // Compute means
    double mean_a = computeMean(image_a);
    double mean_b = computeMean(image_b);
    
    // Compute standard deviations
    double std_a = computeStandardDeviation(image_a, mean_a);
    double std_b = computeStandardDeviation(image_b, mean_b);
    
    // Compute correlation coefficient
    double numerator = 0.0;
    int total_pixels = image_a.rows * image_a.cols;
    
    for (int y = 0; y < image_a.rows; ++y) {
        for (int x = 0; x < image_a.cols; ++x) {
            double val_a = image_a.at<double>(y, x) - mean_a;
            double val_b = image_b.at<double>(y, x) - mean_b;
            numerator += val_a * val_b;
        }
    }
    
    double denominator = std_a * std_b * total_pixels;
    
    if (denominator < 1e-10) {
        return 0.0;
    }
    
    return numerator / denominator;
}

double PearsonCorrelator::computeMean(const cv::Mat& image) {
    if (thread_count_ > 1) {
        // Multi-threaded mean computation
        std::vector<std::thread> threads;
        std::vector<double> partial_sums(thread_count_, 0.0);
        
        int rows_per_thread = image.rows / thread_count_;
        
        for (int t = 0; t < thread_count_; ++t) {
            int start_row = t * rows_per_thread;
            int end_row = (t == thread_count_ - 1) ? image.rows : (t + 1) * rows_per_thread;
            
            threads.emplace_back([this, &image, &partial_sums, t, start_row, end_row]() {
                double sum = 0.0;
                for (int y = start_row; y < end_row; ++y) {
                    for (int x = 0; x < image.cols; ++x) {
                        sum += image.at<double>(y, x);
                    }
                }
                partial_sums[t] = sum;
            });
        }
        
        for (auto& thread : threads) {
            thread.join();
        }
        
        double total_sum = 0.0;
        for (double sum : partial_sums) {
            total_sum += sum;
        }
        
        return total_sum / (image.rows * image.cols);
    } else {
        // Single-threaded mean computation
        cv::Scalar mean = cv::mean(image);
        return mean[0];
    }
}

double PearsonCorrelator::computeStandardDeviation(const cv::Mat& image, double mean) {
    if (thread_count_ > 1) {
        // Multi-threaded standard deviation computation
        std::vector<std::thread> threads;
        std::vector<double> partial_sums(thread_count_, 0.0);
        
        int rows_per_thread = image.rows / thread_count_;
        
        for (int t = 0; t < thread_count_; ++t) {
            int start_row = t * rows_per_thread;
            int end_row = (t == thread_count_ - 1) ? image.rows : (t + 1) * rows_per_thread;
            
            threads.emplace_back([this, &image, &partial_sums, t, start_row, end_row, mean]() {
                double sum_sq = 0.0;
                for (int y = start_row; y < end_row; ++y) {
                    for (int x = 0; x < image.cols; ++x) {
                        double diff = image.at<double>(y, x) - mean;
                        sum_sq += diff * diff;
                    }
                }
                partial_sums[t] = sum_sq;
            });
        }
        
        for (auto& thread : threads) {
            thread.join();
        }
        
        double total_sum_sq = 0.0;
        for (double sum_sq : partial_sums) {
            total_sum_sq += sum_sq;
        }
        
        return std::sqrt(total_sum_sq / (image.rows * image.cols));
    } else {
        // Single-threaded standard deviation computation
        cv::Scalar mean_scalar, stddev;
        cv::meanStdDev(image, mean_scalar, stddev);
        return stddev[0];
    }
}

void PearsonCorrelator::computeMeanAndStd(const cv::Mat& image, double& mean, double& std_dev) {
    if (simd_enabled_) {
        Statistics stats = computeStatisticsBranchless(image);
        mean = stats.mean;
        std_dev = stats.std_dev;
    } else {
        mean = computeMean(image);
        std_dev = computeStandardDeviation(image, mean);
    }
}

cv::Mat PearsonCorrelator::normalizeImage(const cv::Mat& image) {
    if (normalization_type_ == "zscore") {
        return zScoreNormalize(image);
    } else if (normalization_type_ == "minmax") {
        return minMaxNormalize(image);
    } else {
        return image.clone(); // No normalization
    }
}

cv::Mat PearsonCorrelator::zScoreNormalize(const cv::Mat& image) {
    double mean, std_dev;
    computeMeanAndStd(image, mean, std_dev);
    
    cv::Mat normalized = image.clone();
    
    if (std_dev > 1e-10) {
        normalized = (normalized - mean) / std_dev;
    }
    
    return normalized;
}

cv::Mat PearsonCorrelator::minMaxNormalize(const cv::Mat& image) {
    double min_val, max_val;
    cv::minMaxLoc(image, &min_val, &max_val);
    
    cv::Mat normalized = image.clone();
    
    if (max_val > min_val) {
        normalized = (normalized - min_val) / (max_val - min_val);
    }
    
    return normalized;
}

double PearsonCorrelator::computeMeanSIMD(const cv::Mat& image) {
    int total_pixels = image.rows * image.cols;
    const double* data = image.ptr<double>();
    double sum = 0.0;
    
#ifdef USE_AVX512
    int simd_pixels = total_pixels - (total_pixels % 8); // Process 8 doubles at a time
    __m512d sum_vec = _mm512_setzero_pd();
    
    // SIMD computation
    for (int i = 0; i < simd_pixels; i += 8) {
        __m512d data_vec = _mm512_loadu_pd(&data[i]);
        sum_vec = _mm512_add_pd(sum_vec, data_vec);
    }
    
    sum = _mm512_reduce_add_pd(sum_vec);
#elif defined(USE_AVX2) || defined(USE_AVX)
    int simd_pixels = total_pixels - (total_pixels % 4); // Process 4 doubles at a time
    __m256d sum_vec = _mm256_setzero_pd();
    
    // SIMD computation
    for (int i = 0; i < simd_pixels; i += 4) {
        __m256d data_vec = _mm256_loadu_pd(&data[i]);
        sum_vec = _mm256_add_pd(sum_vec, data_vec);
    }
    
    double sum_arr[4];
    _mm256_storeu_pd(sum_arr, sum_vec);
    for (int j = 0; j < 4; ++j) {
        sum += sum_arr[j];
    }
#else
    int simd_pixels = 0;
#endif
    
    // Handle remaining pixels
    for (int i = simd_pixels; i < total_pixels; ++i) {
        sum += data[i];
    }
    
    return sum / total_pixels;
}

double PearsonCorrelator::computeStdDevSIMD(const cv::Mat& image, double mean) {
    int total_pixels = image.rows * image.cols;
    const double* data = image.ptr<double>();
    double sum_sq = 0.0;
    
#ifdef USE_AVX512
    int simd_pixels = total_pixels - (total_pixels % 8);
    __m512d sum_sq_vec = _mm512_setzero_pd();
    __m512d mean_vec = _mm512_set1_pd(mean);
    
    // SIMD computation for the main part
    for (int i = 0; i < simd_pixels; i += 8) {
        __m512d data_vec = _mm512_loadu_pd(&data[i]);
        __m512d diff_vec = _mm512_sub_pd(data_vec, mean_vec);
        __m512d diff_sq_vec = _mm512_mul_pd(diff_vec, diff_vec);
        sum_sq_vec = _mm512_add_pd(sum_sq_vec, diff_sq_vec);
    }
    
    sum_sq = _mm512_reduce_add_pd(sum_sq_vec);
#elif defined(USE_AVX2) || defined(USE_AVX)
    int simd_pixels = total_pixels - (total_pixels % 4);
    __m256d sum_sq_vec = _mm256_setzero_pd();
    __m256d mean_vec = _mm256_set1_pd(mean);
    
    // SIMD computation for the main part
    for (int i = 0; i < simd_pixels; i += 4) {
        __m256d data_vec = _mm256_loadu_pd(&data[i]);
        __m256d diff_vec = _mm256_sub_pd(data_vec, mean_vec);
        __m256d diff_sq_vec = _mm256_mul_pd(diff_vec, diff_vec);
        sum_sq_vec = _mm256_add_pd(sum_sq_vec, diff_sq_vec);
    }
    
    double sum_sq_arr[4];
    _mm256_storeu_pd(sum_sq_arr, sum_sq_vec);
    for (int j = 0; j < 4; ++j) {
        sum_sq += sum_sq_arr[j];
    }
#else
    int simd_pixels = 0;
#endif
    
    // Handle remaining pixels
    for (int i = simd_pixels; i < total_pixels; ++i) {
        double diff = data[i] - mean;
        sum_sq += diff * diff;
    }
    
    return std::sqrt(sum_sq / total_pixels);
}

double PearsonCorrelator::computeCorrelationSIMD(const cv::Mat& image_a, const cv::Mat& image_b) {
    // Compute means using SIMD
    double mean_a = computeMeanSIMD(image_a);
    double mean_b = computeMeanSIMD(image_b);
    
    // Compute standard deviations using SIMD
    double std_a = computeStdDevSIMD(image_a, mean_a);
    double std_b = computeStdDevSIMD(image_b, mean_b);
    
    // Compute correlation coefficient using SIMD
    int total_pixels = image_a.rows * image_a.cols;
    const double* data_a = image_a.ptr<double>();
    const double* data_b = image_b.ptr<double>();
    
    double numerator = 0.0;
    
#ifdef USE_AVX512
    int simd_pixels = total_pixels - (total_pixels % 8);
    __m512d numerator_vec = _mm512_setzero_pd();
    __m512d mean_a_vec = _mm512_set1_pd(mean_a);
    __m512d mean_b_vec = _mm512_set1_pd(mean_b);
    
    // SIMD computation for the main part
    for (int i = 0; i < simd_pixels; i += 8) {
        __m512d vec_a = _mm512_loadu_pd(&data_a[i]);
        __m512d vec_b = _mm512_loadu_pd(&data_b[i]);
        
        __m512d diff_a = _mm512_sub_pd(vec_a, mean_a_vec);
        __m512d diff_b = _mm512_sub_pd(vec_b, mean_b_vec);
        
        __m512d product = _mm512_mul_pd(diff_a, diff_b);
        numerator_vec = _mm512_add_pd(numerator_vec, product);
    }
    
    numerator = _mm512_reduce_add_pd(numerator_vec);
#elif defined(USE_AVX2) || defined(USE_AVX)
    int simd_pixels = total_pixels - (total_pixels % 4);
    __m256d numerator_vec = _mm256_setzero_pd();
    __m256d mean_a_vec = _mm256_set1_pd(mean_a);
    __m256d mean_b_vec = _mm256_set1_pd(mean_b);
    
    // SIMD computation for the main part
    for (int i = 0; i < simd_pixels; i += 4) {
        __m256d vec_a = _mm256_loadu_pd(&data_a[i]);
        __m256d vec_b = _mm256_loadu_pd(&data_b[i]);
        
        __m256d diff_a = _mm256_sub_pd(vec_a, mean_a_vec);
        __m256d diff_b = _mm256_sub_pd(vec_b, mean_b_vec);
        
        __m256d product = _mm256_mul_pd(diff_a, diff_b);
        numerator_vec = _mm256_add_pd(numerator_vec, product);
    }
    
    double numerator_arr[4];
    _mm256_storeu_pd(numerator_arr, numerator_vec);
    for (int j = 0; j < 4; ++j) {
        numerator += numerator_arr[j];
    }
#else
    int simd_pixels = 0;
#endif
    
    // Handle remaining pixels
    for (int i = simd_pixels; i < total_pixels; ++i) {
        double val_a = data_a[i] - mean_a;
        double val_b = data_b[i] - mean_b;
        numerator += val_a * val_b;
    }
    
    double denominator = std_a * std_b * total_pixels;
    
    if (denominator < 1e-10) {
        return 0.0;
    }
    
    return numerator / denominator;
}

cv::Mat PearsonCorrelator::flattenImage(const cv::Mat& image) {
    return image.reshape(1, image.rows * image.cols);
}

bool PearsonCorrelator::validateImageDimensions(const cv::Mat& image_a, const cv::Mat& image_b) {
    return image_a.size() == image_b.size() && image_a.type() == image_b.type();
}

PearsonCorrelator::Statistics PearsonCorrelator::computeStatisticsBranchless(const cv::Mat& image) {
    Statistics stats;
    int total_pixels = image.rows * image.cols;
    
    // Compute sum and sum of squares in a single pass
    for (int y = 0; y < image.rows; ++y) {
        for (int x = 0; x < image.cols; ++x) {
            double val = image.at<double>(y, x);
            stats.sum += val;
            stats.sum_sq += val * val;
        }
    }
    
    stats.count = total_pixels;
    stats.mean = stats.sum / stats.count;
    
    // Compute standard deviation
    double variance = (stats.sum_sq / stats.count) - (stats.mean * stats.mean);
    stats.std_dev = std::sqrt(std::max(0.0, variance));
    
    return stats;
}

void PearsonCorrelator::resizeTempBuffer(size_t size) {
    if (temp_buffer_.size() < size) {
        temp_buffer_.resize(size);
    }
}

void PearsonCorrelator::setThreadCount(int thread_count) {
    thread_count_ = thread_count;
    if (thread_manager_) {
        thread_manager_->setThreadCount(thread_count);
    }
}

void PearsonCorrelator::setNormalizationType(const std::string& type) {
    if (type == "zscore" || type == "minmax" || type == "none") {
        normalization_type_ = type;
    } else {
        throw std::invalid_argument("Invalid normalization type. Use 'zscore', 'minmax', or 'none'");
    }
}

void PearsonCorrelator::enableSIMD(bool enable) {
    simd_enabled_ = enable;
}

void PearsonCorrelator::setMemoryPool(std::shared_ptr<MemoryPool> pool) {

}

} // namespace imgreg 
#pragma once

#include <vector>
#include <string>
#include <chrono>
#include <complex>

namespace imgreg {

struct CorrelationResult {
    // Correlation method used
    std::string method_name;
    
    // Correlation value (peak correlation)
    double correlation_value;
    
    // Alignment offset (x, y coordinates)
    int offset_x;
    int offset_y;
    
    // Perf metrics
    std::chrono::microseconds execution_time;
    size_t memory_usage_bytes;
    int thread_count;
    double cpu_utilization;
    
    // FFT-specific data
    std::vector<std::vector<std::complex<double>>> frequency_domain;
    
    // Spatial method specific data
    std::vector<std::vector<double>> correlation_map;
    
    // Pearson method specific data
    double pearson_coefficient;
    
    // Constructor
    CorrelationResult() : 
        correlation_value(0.0), 
        offset_x(0), 
        offset_y(0), 
        memory_usage_bytes(0), 
        thread_count(1), 
        cpu_utilization(0.0),
        pearson_coefficient(0.0) {}
};

struct PerformanceMetrics {
    // Timing metrics
    std::chrono::microseconds total_time;
    std::chrono::microseconds fft_time;
    std::chrono::microseconds spatial_time;
    std::chrono::microseconds pearson_time;
    
    // Memory metrics
    size_t peak_memory_usage;
    size_t average_memory_usage;
    
    // Threading metrics
    int optimal_thread_count;
    double thread_efficiency;
    double speedup_ratio;
    
    // Algorithm-specific metrics
    double flops_per_second;
    double memory_bandwidth_gbps;
    
    // Constructor
    PerformanceMetrics() : 
        total_time(std::chrono::microseconds(0)),
        fft_time(std::chrono::microseconds(0)),
        spatial_time(std::chrono::microseconds(0)),
        pearson_time(std::chrono::microseconds(0)),
        peak_memory_usage(0),
        average_memory_usage(0),
        optimal_thread_count(1),
        thread_efficiency(0.0),
        speedup_ratio(1.0),
        flops_per_second(0.0),
        memory_bandwidth_gbps(0.0) {}
};

} // namespace imgreg 
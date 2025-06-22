#include "performance_analyzer.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <map>
#include <numeric>
#include <cmath>

namespace imgreg {

PerformanceAnalyzer::PerformanceAnalyzer()
    : detailed_logging_(false),
      benchmark_mode_(false),
      total_operations_(0),
      successful_operations_(0),
      failed_operations_(0) {
}

PerformanceAnalyzer::~PerformanceAnalyzer() = default;

void PerformanceAnalyzer::recordResult(const CorrelationResult& result) {
    results_.push_back(result);
    
    // Update method statistics
    MethodStats& stats = getOrCreateMethodStats(result.method_name);
    stats.execution_times.push_back(result.execution_time);
    stats.memory_usage.push_back(result.memory_usage_bytes);
    stats.thread_counts.push_back(result.thread_count);
    stats.correlation_values.push_back(result.correlation_value);
    
    total_operations_++;
    if (result.correlation_value > 0) {
        successful_operations_++;
    } else {
        failed_operations_++;
    }
    
    // Record memory usage over time
    memory_timeline_.emplace_back(std::chrono::high_resolution_clock::now(), result.memory_usage_bytes);
}

void PerformanceAnalyzer::recordMetrics(const PerformanceMetrics& metrics) {
    metrics_.push_back(metrics);
}

void PerformanceAnalyzer::startSession() {
    session_start_ = std::chrono::high_resolution_clock::now();
    results_.clear();
    metrics_.clear();
    method_statistics_.clear();
    memory_timeline_.clear();
    total_operations_ = 0;
    successful_operations_ = 0;
    failed_operations_ = 0;
}

void PerformanceAnalyzer::endSession() {
    session_end_ = std::chrono::high_resolution_clock::now();
}

PerformanceMetrics PerformanceAnalyzer::analyzePerformance() const {
    PerformanceMetrics metrics;
    
    if (results_.empty()) {
        return metrics;
    }
    
    // Calculate timing metrics
    auto total_duration = std::chrono::duration_cast<std::chrono::microseconds>(session_end_ - session_start_);
    metrics.total_time = total_duration;
    
    // Calculate method-specific times
    for (const auto& result : results_) {
        if (result.method_name == "fft") {
            metrics.fft_time += result.execution_time;
        } else if (result.method_name == "spatial") {
            metrics.spatial_time += result.execution_time;
        } else if (result.method_name == "pearson") {
            metrics.pearson_time += result.execution_time;
        }
    }
    
    // Calculate memory metrics
    size_t peak_memory = 0;
    size_t total_memory = 0;
    for (const auto& result : results_) {
        peak_memory = std::max(peak_memory, result.memory_usage_bytes);
        total_memory += result.memory_usage_bytes;
    }
    
    metrics.peak_memory_usage = peak_memory;
    metrics.average_memory_usage = total_memory / results_.size();
    
    // Calculate threading metrics
    int max_threads = 0;
    double total_thread_utilization = 0.0;
    for (const auto& result : results_) {
        max_threads = std::max(max_threads, result.thread_count);
        total_thread_utilization += result.cpu_utilization;
    }
    
    metrics.optimal_thread_count = max_threads;
    metrics.thread_efficiency = total_thread_utilization / results_.size();
    metrics.speedup_ratio = metrics.thread_efficiency * max_threads;
    
    return metrics;
}

std::vector<CorrelationResult> PerformanceAnalyzer::getBestResults() const {
    std::vector<CorrelationResult> best_results;
    
    // Group results by method
    std::map<std::string, std::vector<CorrelationResult>> method_results;
    for (const auto& result : results_) {
        method_results[result.method_name].push_back(result);
    }
    
    // Find best result for each method
    for (const auto& [method, method_result_list] : method_results) {
        auto best_it = std::max_element(method_result_list.begin(), method_result_list.end(),
            [](const CorrelationResult& a, const CorrelationResult& b) {
                return a.correlation_value < b.correlation_value;
            });
        
        if (best_it != method_result_list.end()) {
            best_results.push_back(*best_it);
        }
    }
    
    return best_results;
}

std::string PerformanceAnalyzer::generateReport() const {
    std::stringstream report;
    
    report << "=== Image Correlation Performance Report ===\n\n";
    
    // Session information
    auto session_duration = std::chrono::duration_cast<std::chrono::microseconds>(session_end_ - session_start_);
    report << "Session Duration: " << session_duration.count() << " μs\n";
    report << "Total Operations: " << total_operations_ << "\n";
    report << "Successful Operations: " << successful_operations_ << "\n";
    report << "Failed Operations: " << failed_operations_ << "\n";
    report << "Success Rate: " << (total_operations_ > 0 ? (100.0 * successful_operations_ / total_operations_) : 0.0) << "%\n\n";
    
    // Method comparison
    report << "=== Method Comparison ===\n";
    for (const auto& stats : method_statistics_) {
        if (stats.execution_times.empty()) continue;
        
        report << "\nMethod: " << stats.method_name << "\n";
        
        // Timing statistics
        auto avg_time = std::accumulate(stats.execution_times.begin(), stats.execution_times.end(), 
                                       std::chrono::microseconds(0)) / stats.execution_times.size();
        report << "  Average Execution Time: " << avg_time.count() << " μs\n";
        
        // Memory statistics
        size_t avg_memory = std::accumulate(stats.memory_usage.begin(), stats.memory_usage.end(), 0ULL) / stats.memory_usage.size();
        report << "  Average Memory Usage: " << avg_memory << " bytes\n";
        
        // Correlation statistics
        double avg_correlation = std::accumulate(stats.correlation_values.begin(), stats.correlation_values.end(), 0.0) / stats.correlation_values.size();
        report << "  Average Correlation Value: " << std::fixed << std::setprecision(6) << avg_correlation << "\n";
        
        // Threading statistics
        int avg_threads = std::accumulate(stats.thread_counts.begin(), stats.thread_counts.end(), 0) / stats.thread_counts.size();
        report << "  Average Thread Count: " << avg_threads << "\n";
    }
    
    // Performance metrics
    PerformanceMetrics metrics = analyzePerformance();
    report << "\n=== Performance Metrics ===\n";
    report << "Total Time: " << metrics.total_time.count() << " μs\n";
    report << "FFT Time: " << metrics.fft_time.count() << " μs\n";
    report << "Spatial Time: " << metrics.spatial_time.count() << " μs\n";
    report << "Pearson Time: " << metrics.pearson_time.count() << " μs\n";
    report << "Peak Memory Usage: " << metrics.peak_memory_usage << " bytes\n";
    report << "Average Memory Usage: " << metrics.average_memory_usage << " bytes\n";
    report << "Optimal Thread Count: " << metrics.optimal_thread_count << "\n";
    report << "Thread Efficiency: " << std::fixed << std::setprecision(2) << (metrics.thread_efficiency * 100) << "%\n";
    report << "Speedup Ratio: " << std::fixed << std::setprecision(2) << metrics.speedup_ratio << "x\n";
    
    return report.str();
}

double PerformanceAnalyzer::computeAverageTime(const std::string& method) const {
    auto it = std::find_if(method_statistics_.begin(), method_statistics_.end(),
        [&method](const MethodStats& stats) { return stats.method_name == method; });
    
    if (it == method_statistics_.end() || it->execution_times.empty()) {
        return 0.0;
    }
    
    auto total_time = std::accumulate(it->execution_times.begin(), it->execution_times.end(), 
                                     std::chrono::microseconds(0));
    return static_cast<double>(total_time.count()) / it->execution_times.size();
}

double PerformanceAnalyzer::computeStandardDeviation(const std::string& method) const {
    auto it = std::find_if(method_statistics_.begin(), method_statistics_.end(),
        [&method](const MethodStats& stats) { return stats.method_name == method; });
    
    if (it == method_statistics_.end() || it->execution_times.size() < 2) {
        return 0.0;
    }
    
    double mean = computeAverageTime(method);
    double sum_sq = 0.0;
    
    for (const auto& time : it->execution_times) {
        double diff = static_cast<double>(time.count()) - mean;
        sum_sq += diff * diff;
    }
    
    return std::sqrt(sum_sq / (it->execution_times.size() - 1));
}

double PerformanceAnalyzer::computeSpeedupRatio(const std::string& method1, const std::string& method2) const {
    double time1 = computeAverageTime(method1);
    double time2 = computeAverageTime(method2);
    
    if (time2 == 0.0) {
        return 0.0;
    }
    
    return time1 / time2;
}

int PerformanceAnalyzer::findOptimalThreadCount(const std::string& method) const {
    auto it = std::find_if(method_statistics_.begin(), method_statistics_.end(),
        [&method](const MethodStats& stats) { return stats.method_name == method; });
    
    if (it == method_statistics_.end() || it->thread_counts.empty()) {
        return 1;
    }
    
    // Find thread count with best performance (lowest average time)
    std::map<int, std::vector<std::chrono::microseconds>> thread_times;
    for (size_t i = 0; i < it->thread_counts.size(); ++i) {
        thread_times[it->thread_counts[i]].push_back(it->execution_times[i]);
    }
    
    int optimal_threads = 1;
    double best_avg_time = std::numeric_limits<double>::max();
    
    for (const auto& [thread_count, times] : thread_times) {
        double avg_time = std::accumulate(times.begin(), times.end(), std::chrono::microseconds(0)).count() / times.size();
        if (avg_time < best_avg_time) {
            best_avg_time = avg_time;
            optimal_threads = thread_count;
        }
    }
    
    return optimal_threads;
}

double PerformanceAnalyzer::computeThreadEfficiency(const std::string& method) const {
    auto it = std::find_if(method_statistics_.begin(), method_statistics_.end(),
        [&method](const MethodStats& stats) { return stats.method_name == method; });
    
    if (it == method_statistics_.end() || it->thread_counts.empty()) {
        return 0.0;
    }
    
    // Calculate average thread utilization
    double total_utilization = 0.0;
    int count = 0;
    
    for (const auto& result : results_) {
        if (result.method_name == method) {
            total_utilization += result.cpu_utilization;
            count++;
        }
    }
    
    return count > 0 ? total_utilization / count : 0.0;
}

double PerformanceAnalyzer::computeScalability(const std::string& method) const {
    // Calculate how well the method scales with thread count
    auto it = std::find_if(method_statistics_.begin(), method_statistics_.end(),
        [&method](const MethodStats& stats) { return stats.method_name == method; });
    
    if (it == method_statistics_.end() || it->thread_counts.size() < 2) {
        return 0.0;
    }
    
    // Find single-threaded and multi-threaded performance
    double single_thread_time = std::numeric_limits<double>::max();
    double multi_thread_time = std::numeric_limits<double>::max();
    
    for (size_t i = 0; i < it->thread_counts.size(); ++i) {
        double time = static_cast<double>(it->execution_times[i].count());
        if (it->thread_counts[i] == 1) {
            single_thread_time = std::min(single_thread_time, time);
        } else {
            multi_thread_time = std::min(multi_thread_time, time);
        }
    }
    
    if (single_thread_time == std::numeric_limits<double>::max() || multi_thread_time == std::numeric_limits<double>::max()) {
        return 0.0;
    }
    
    return single_thread_time / multi_thread_time;
}

size_t PerformanceAnalyzer::getPeakMemoryUsage() const {
    size_t peak = 0;
    for (const auto& result : results_) {
        peak = std::max(peak, result.memory_usage_bytes);
    }
    return peak;
}

double PerformanceAnalyzer::computeMemoryEfficiency() const {
    if (results_.empty()) {
        return 0.0;
    }
    
    size_t total_memory = 0;
    for (const auto& result : results_) {
        total_memory += result.memory_usage_bytes;
    }
    
    return static_cast<double>(total_memory) / results_.size();
}

std::vector<size_t> PerformanceAnalyzer::getMemoryUsageOverTime() const {
    std::vector<size_t> usage;
    for (const auto& [time, memory] : memory_timeline_) {
        usage.push_back(memory);
    }
    return usage;
}

void PerformanceAnalyzer::exportToCSV(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }
    
    writeCSVHeader(file);
    for (const auto& result : results_) {
        writeCSVRow(file, result);
    }
    
    file.close();
}

void PerformanceAnalyzer::exportToJSON(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }
    
    file << "{\n";
    file << "  \"session\": {\n";
    file << "    \"start_time\": \"" << session_start_.time_since_epoch().count() << "\",\n";
    file << "    \"end_time\": \"" << session_end_.time_since_epoch().count() << "\",\n";
    file << "    \"total_operations\": " << total_operations_ << ",\n";
    file << "    \"successful_operations\": " << successful_operations_ << ",\n";
    file << "    \"failed_operations\": " << failed_operations_ << "\n";
    file << "  },\n";
    file << "  \"results\": [\n";
    
    for (size_t i = 0; i < results_.size(); ++i) {
        file << "    " << resultToJSON(results_[i]);
        if (i < results_.size() - 1) {
            file << ",";
        }
        file << "\n";
    }
    
    file << "  ]\n";
    file << "}\n";
    
    file.close();
}

void PerformanceAnalyzer::generateCharts(const std::string& output_dir) const {
    std::ofstream timing_file(output_dir + "/timing_chart.txt");
    timing_file << "Timing Chart Data\n";
    timing_file.close();
    
    std::ofstream memory_file(output_dir + "/memory_chart.txt");
    memory_file << "Memory Chart Data\n";
    memory_file.close();
}

void PerformanceAnalyzer::setOutputDirectory(const std::string& dir) {
    output_directory_ = dir;
}

void PerformanceAnalyzer::enableDetailedLogging(bool enable) {
    detailed_logging_ = enable;
}

void PerformanceAnalyzer::setBenchmarkMode(bool enable) {
    benchmark_mode_ = enable;
}

PerformanceAnalyzer::MethodStats& PerformanceAnalyzer::getOrCreateMethodStats(const std::string& method_name) {
    auto it = std::find_if(method_statistics_.begin(), method_statistics_.end(),
        [&method_name](const MethodStats& stats) { return stats.method_name == method_name; });
    
    if (it == method_statistics_.end()) {
        method_statistics_.emplace_back(method_name);
        return method_statistics_.back();
    }
    
    return *it;
}

void PerformanceAnalyzer::updateMethodStats(const CorrelationResult& result) {
    MethodStats& stats = getOrCreateMethodStats(result.method_name);
    stats.execution_times.push_back(result.execution_time);
    stats.memory_usage.push_back(result.memory_usage_bytes);
    stats.thread_counts.push_back(result.thread_count);
    stats.correlation_values.push_back(result.correlation_value);
}

double PerformanceAnalyzer::computePercentile(const std::vector<double>& data, double percentile) const {
    if (data.empty()) {
        return 0.0;
    }
    
    std::vector<double> sorted_data = data;
    std::sort(sorted_data.begin(), sorted_data.end());
    
    size_t index = static_cast<size_t>(percentile * sorted_data.size() / 100.0);
    if (index >= sorted_data.size()) {
        index = sorted_data.size() - 1;
    }
    
    return sorted_data[index];
}

void PerformanceAnalyzer::writeCSVHeader(std::ofstream& file) const {
    file << "Method,CorrelationValue,OffsetX,OffsetY,ExecutionTime,MemoryUsage,ThreadCount,CPUUtilization\n";
}

void PerformanceAnalyzer::writeCSVRow(std::ofstream& file, const CorrelationResult& result) const {
    file << result.method_name << ","
         << result.correlation_value << ","
         << result.offset_x << ","
         << result.offset_y << ","
         << result.execution_time.count() << ","
         << result.memory_usage_bytes << ","
         << result.thread_count << ","
         << result.cpu_utilization << "\n";
}

std::string PerformanceAnalyzer::resultToJSON(const CorrelationResult& result) const {
    std::stringstream json;
    json << "{";
    json << "\"method\": \"" << result.method_name << "\",";
    json << "\"correlation_value\": " << result.correlation_value << ",";
    json << "\"offset_x\": " << result.offset_x << ",";
    json << "\"offset_y\": " << result.offset_y << ",";
    json << "\"execution_time\": " << result.execution_time.count() << ",";
    json << "\"memory_usage\": " << result.memory_usage_bytes << ",";
    json << "\"thread_count\": " << result.thread_count << ",";
    json << "\"cpu_utilization\": " << result.cpu_utilization;
    json << "}";
    return json.str();
}

} // namespace imgreg 
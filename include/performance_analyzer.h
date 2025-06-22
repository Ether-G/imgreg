#pragma once

#include "correlation_result.h"
#include <vector>
#include <string>
#include <chrono>
#include <memory>
#include <fstream>
#include <functional>
#include <thread>
#include <atomic>

namespace imgreg {

class PerformanceAnalyzer {
public:
    PerformanceAnalyzer();
    ~PerformanceAnalyzer();
    
    // Data collection
    void recordResult(const CorrelationResult& result);
    void recordMetrics(const PerformanceMetrics& metrics);
    void startSession();
    void endSession();
    
    // Analysis methods
    PerformanceMetrics analyzePerformance() const;
    std::vector<CorrelationResult> getBestResults() const;
    std::string generateReport() const;
    
    // Statistical analysis
    double computeAverageTime(const std::string& method) const;
    double computeStandardDeviation(const std::string& method) const;
    double computeSpeedupRatio(const std::string& method1, const std::string& method2) const;
    
    // Threading analysis
    int findOptimalThreadCount(const std::string& method) const;
    double computeThreadEfficiency(const std::string& method) const;
    double computeScalability(const std::string& method) const;
    
    // Memory analysis
    size_t getPeakMemoryUsage() const;
    double computeMemoryEfficiency() const;
    std::vector<size_t> getMemoryUsageOverTime() const;
    
    // Export functionality
    void exportToCSV(const std::string& filename) const;
    void exportToJSON(const std::string& filename) const;
    void generateCharts(const std::string& output_dir) const;
    
    // Configuration
    void setOutputDirectory(const std::string& dir);
    void enableDetailedLogging(bool enable);
    void setBenchmarkMode(bool enable);

private:
    // Data storage
    std::vector<CorrelationResult> results_;
    std::vector<PerformanceMetrics> metrics_;
    
    // Session tracking
    std::chrono::high_resolution_clock::time_point session_start_;
    std::chrono::high_resolution_clock::time_point session_end_;
    
    // Configuration
    std::string output_directory_;
    bool detailed_logging_;
    bool benchmark_mode_;
    
    // Analysis helpers
    struct MethodStats {
        std::string method_name;
        std::vector<std::chrono::microseconds> execution_times;
        std::vector<size_t> memory_usage;
        std::vector<int> thread_counts;
        std::vector<double> correlation_values;
        
        MethodStats() = default;
        explicit MethodStats(const std::string& name) : method_name(name) {}
    };
    
    std::vector<MethodStats> method_statistics_;
    
    // Helper methods
    MethodStats& getOrCreateMethodStats(const std::string& method_name);
    void updateMethodStats(const CorrelationResult& result);
    double computePercentile(const std::vector<double>& data, double percentile) const;
    
    // Chart generation
    void generateTimingChart(const std::string& filename) const;
    void generateMemoryChart(const std::string& filename) const;
    void generateThreadingChart(const std::string& filename) const;
    void generateCorrelationChart(const std::string& filename) const;
    
    // CSV/JSON export helpers
    void writeCSVHeader(std::ofstream& file) const;
    void writeCSVRow(std::ofstream& file, const CorrelationResult& result) const;
    std::string resultToJSON(const CorrelationResult& result) const;
    
    // Performance counters
    size_t total_operations_;
    size_t successful_operations_;
    size_t failed_operations_;
    
    // Memory tracking
    std::vector<std::pair<std::chrono::high_resolution_clock::time_point, size_t>> memory_timeline_;
};

// Utility class for real-time performance monitoring
class PerformanceMonitor {
public:
    PerformanceMonitor();
    ~PerformanceMonitor();
    
    // Monitoring control
    void startMonitoring();
    void stopMonitoring();
    void pauseMonitoring();
    void resumeMonitoring();
    
    // Real-time metrics
    double getCurrentCPUUsage() const;
    size_t getCurrentMemoryUsage() const;
    int getCurrentThreadCount() const;
    
    // Alert system
    void setMemoryThreshold(size_t threshold);
    void setCPUThreshold(double threshold);
    void setAlertCallback(std::function<void(const std::string&)> callback);

private:
    // Monitoring state
    bool monitoring_active_;
    bool monitoring_paused_;
    
    // Thresholds
    size_t memory_threshold_;
    double cpu_threshold_;
    
    // Callbacks
    std::function<void(const std::string&)> alert_callback_;
    
    // System monitoring
    void monitorSystem();
    double getSystemCPUUsage();
    size_t getSystemMemoryUsage();
    
    // Thread for monitoring
    std::thread monitor_thread_;
    std::atomic<bool> stop_monitoring_;
};

} // namespace imgreg 
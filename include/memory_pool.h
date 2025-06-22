#pragma once

#include <vector>
#include <mutex>
#include <memory>
#include <unordered_map>
#include <cstddef>
#include <atomic>

namespace imgreg {

// Memory block structure
struct MemoryBlock {
    void* ptr;
    size_t size;
    bool in_use;
    
    MemoryBlock() : ptr(nullptr), size(0), in_use(false) {}
    MemoryBlock(void* p, size_t s) : ptr(p), size(s), in_use(false) {}
};

// Memory pool for efficient allocation
class MemoryPool {
public:
    explicit MemoryPool(size_t initial_size = 1024 * 1024); // 1MB default
    ~MemoryPool();
    
    // Memory allocation
    void* allocate(size_t size);
    void* allocateAligned(size_t size, size_t alignment);
    
    // Memory deallocation
    void deallocate(void* ptr);
    
    // Pool management
    void reserve(size_t size);
    void clear();
    void shrink();
    
    // Statistics
    size_t getTotalAllocated() const;
    size_t getTotalReserved() const;
    size_t getPeakUsage() const;
    double getFragmentationRatio() const;
    
    // Configuration
    void setGrowthFactor(double factor);
    void setMaxSize(size_t max_size);

private:
    // Memory management
    void* allocateBlock(size_t size);
    void* allocateAlignedBlock(size_t size, size_t alignment);
    void deallocateBlock(void* ptr);
    
    // Block management
    MemoryBlock* findFreeBlock(size_t size);
    MemoryBlock* findAlignedBlock(size_t size, size_t alignment);
    void mergeAdjacentBlocks();
    
    // Memory allocation helpers
    void* allocateFromOS(size_t size);
    void* allocateAlignedFromOS(size_t size, size_t alignment);
    void deallocateToOS(void* ptr, size_t size);
    
    // Pool state
    std::vector<MemoryBlock> blocks_;
    std::vector<void*> os_allocations_;
    
    // Configuration
    size_t initial_size_;
    size_t max_size_;
    double growth_factor_;
    
    // Statistics
    size_t total_allocated_;
    size_t total_reserved_;
    size_t peak_usage_;
    
    // Synchronization
    mutable std::mutex pool_mutex_;
    
    // Alignment constants
    static constexpr size_t DEFAULT_ALIGNMENT = 64; // Cache line size
    static constexpr size_t MIN_BLOCK_SIZE = 16;
};

// Thread-local memory pool
class ThreadLocalMemoryPool {
public:
    ThreadLocalMemoryPool();
    ~ThreadLocalMemoryPool();
    
    // Thread-local allocation
    void* allocate(size_t size);
    void* allocateAligned(size_t size, size_t alignment);
    void deallocate(void* ptr);
    
    // Pool management
    void clear();
    
    // Statistics
    size_t getUsage() const;

private:
    std::unique_ptr<MemoryPool> local_pool_;
    static thread_local ThreadLocalMemoryPool* instance_;
    
    // Thread-local storage
    static void cleanupThreadLocal();
};

// Global memory manager
class MemoryManager {
public:
    static MemoryManager& getInstance();
    
    // Global allocation
    void* allocate(size_t size);
    void* allocateAligned(size_t size, size_t alignment);
    void deallocate(void* ptr);
    
    // Pool management
    void setGlobalPoolSize(size_t size);
    void setThreadLocalPoolSize(size_t size);
    
    // Statistics
    void printStatistics() const;
    void resetStatistics();

private:
    MemoryManager();
    ~MemoryManager();
    
    std::unique_ptr<MemoryPool> global_pool_;
    size_t thread_local_pool_size_;
    
    // Statistics
    std::atomic<size_t> total_allocations_;
    std::atomic<size_t> total_deallocations_;
    std::atomic<size_t> peak_memory_usage_;
};

} // namespace imgreg 
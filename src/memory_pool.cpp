#include "memory_pool.h"
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <iostream>

namespace imgreg {

MemoryPool::MemoryPool(size_t initial_size)
    : initial_size_(initial_size),
      max_size_(initial_size * 10),
      growth_factor_(2.0),
      total_allocated_(0),
      total_reserved_(0),
      peak_usage_(0) {
    
    // Allocate initial memory
    reserve(initial_size);
}

MemoryPool::~MemoryPool() {
    clear();
}

void* MemoryPool::allocate(size_t size) {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    
    // Find a suitable block
    MemoryBlock* block = findFreeBlock(size);
    if (block) {
        block->in_use = true;
        total_allocated_ += size;
        peak_usage_ = std::max(peak_usage_, total_allocated_);
        return block->ptr;
    }
    
    // Allocate new block from OS
    void* ptr = allocateFromOS(size);
    if (ptr) {
        blocks_.emplace_back(ptr, size);
        blocks_.back().in_use = true;
        total_allocated_ += size;
        total_reserved_ += size;
        peak_usage_ = std::max(peak_usage_, total_allocated_);
    }
    
    return ptr;
}

void* MemoryPool::allocateAligned(size_t size, size_t alignment) {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    
    // Find a suitable aligned block
    MemoryBlock* block = findAlignedBlock(size, alignment);
    if (block) {
        block->in_use = true;
        total_allocated_ += size;
        peak_usage_ = std::max(peak_usage_, total_allocated_);
        return block->ptr;
    }
    
    // Allocate new aligned block from OS
    void* ptr = allocateAlignedFromOS(size, alignment);
    if (ptr) {
        blocks_.emplace_back(ptr, size);
        blocks_.back().in_use = true;
        total_allocated_ += size;
        total_reserved_ += size;
        peak_usage_ = std::max(peak_usage_, total_allocated_);
    }
    
    return ptr;
}

void MemoryPool::deallocate(void* ptr) {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    
    // Find the block containing this pointer
    for (auto& block : blocks_) {
        if (block.ptr == ptr) {
            block.in_use = false;
            total_allocated_ -= block.size;
            return;
        }
    }
    
    // If not found in pool, deallocate directly
    deallocateToOS(ptr, 0);
}

void MemoryPool::reserve(size_t size) {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    
    if (total_reserved_ < size) {
        size_t additional = size - total_reserved_;
        void* ptr = allocateFromOS(additional);
        if (ptr) {
            blocks_.emplace_back(ptr, additional);
            total_reserved_ += additional;
        }
    }
}

void MemoryPool::clear() {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    
    for (auto& block : blocks_) {
        if (block.ptr) {
            deallocateToOS(block.ptr, block.size);
        }
    }
    
    blocks_.clear();
    total_allocated_ = 0;
    total_reserved_ = 0;
    peak_usage_ = 0;
}

void MemoryPool::shrink() {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    
    // Remove unused blocks
    blocks_.erase(
        std::remove_if(blocks_.begin(), blocks_.end(),
            [this](const MemoryBlock& block) {
                if (!block.in_use) {
                    deallocateToOS(block.ptr, block.size);
                    total_reserved_ -= block.size;
                    return true;
                }
                return false;
            }),
        blocks_.end()
    );
    
    // Merge adjacent free blocks
    mergeAdjacentBlocks();
}

size_t MemoryPool::getTotalAllocated() const {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    return total_allocated_;
}

size_t MemoryPool::getTotalReserved() const {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    return total_reserved_;
}

size_t MemoryPool::getPeakUsage() const {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    return peak_usage_;
}

double MemoryPool::getFragmentationRatio() const {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    
    if (total_reserved_ == 0) {
        return 0.0;
    }
    
    return 1.0 - (static_cast<double>(total_allocated_) / total_reserved_);
}

void MemoryPool::setGrowthFactor(double factor) {
    growth_factor_ = factor;
}

void MemoryPool::setMaxSize(size_t max_size) {
    max_size_ = max_size;
}

MemoryBlock* MemoryPool::findFreeBlock(size_t size) {
    for (auto& block : blocks_) {
        if (!block.in_use && block.size >= size) {
            return &block;
        }
    }
    return nullptr;
}

MemoryBlock* MemoryPool::findAlignedBlock(size_t size, size_t alignment) {
    for (auto& block : blocks_) {
        if (!block.in_use && block.size >= size) {
            // Check if the block is properly aligned
            uintptr_t addr = reinterpret_cast<uintptr_t>(block.ptr);
            if (addr % alignment == 0) {
                return &block;
            }
        }
    }
    return nullptr;
}

void MemoryPool::mergeAdjacentBlocks() {

}

void* MemoryPool::allocateFromOS(size_t size) {
    return std::malloc(size);
}

void* MemoryPool::allocateAlignedFromOS(size_t size, size_t alignment) {
    void* ptr = nullptr;
    if (posix_memalign(&ptr, alignment, size) != 0) {
        return nullptr;
    }
    return ptr;
}

void MemoryPool::deallocateToOS(void* ptr, size_t size) {
    std::free(ptr);
}

// ThreadLocalMemoryPool implementation
thread_local ThreadLocalMemoryPool* ThreadLocalMemoryPool::instance_ = nullptr;

ThreadLocalMemoryPool::ThreadLocalMemoryPool()
    : local_pool_(std::make_unique<MemoryPool>(1024 * 1024)) {
}

ThreadLocalMemoryPool::~ThreadLocalMemoryPool() {
    cleanupThreadLocal();
}

void* ThreadLocalMemoryPool::allocate(size_t size) {
    if (!instance_) {
        instance_ = new ThreadLocalMemoryPool();
    }
    return instance_->local_pool_->allocate(size);
}

void* ThreadLocalMemoryPool::allocateAligned(size_t size, size_t alignment) {
    if (!instance_) {
        instance_ = new ThreadLocalMemoryPool();
    }
    return instance_->local_pool_->allocateAligned(size, alignment);
}

void ThreadLocalMemoryPool::deallocate(void* ptr) {
    if (instance_) {
        instance_->local_pool_->deallocate(ptr);
    }
}

void ThreadLocalMemoryPool::clear() {
    if (instance_) {
        instance_->local_pool_->clear();
    }
}

size_t ThreadLocalMemoryPool::getUsage() const {
    return local_pool_->getTotalAllocated();
}

void ThreadLocalMemoryPool::cleanupThreadLocal() {
    if (instance_) {
        delete instance_;
        instance_ = nullptr;
    }
}

// MemoryManager implementation
MemoryManager& MemoryManager::getInstance() {
    static MemoryManager instance;
    return instance;
}

MemoryManager::MemoryManager()
    : thread_local_pool_size_(1024 * 1024),
      total_allocations_(0),
      total_deallocations_(0),
      peak_memory_usage_(0) {
    
    global_pool_ = std::make_unique<MemoryPool>(10 * 1024 * 1024); // 10MB initial
}

MemoryManager::~MemoryManager() = default;

void* MemoryManager::allocate(size_t size) {
    total_allocations_++;
    void* ptr = global_pool_->allocate(size);
    if (ptr) {
        size_t current_usage = global_pool_->getTotalAllocated();
        // Update peak memory usage atomically
        size_t current_peak = peak_memory_usage_.load();
        while (current_usage > current_peak && 
               !peak_memory_usage_.compare_exchange_weak(current_peak, current_usage)) {
            // Retry if compare_exchange_weak failed
        }
    }
    return ptr;
}

void* MemoryManager::allocateAligned(size_t size, size_t alignment) {
    total_allocations_++;
    void* ptr = global_pool_->allocateAligned(size, alignment);
    if (ptr) {
        size_t current_usage = global_pool_->getTotalAllocated();
        // Update peak memory usage atomically
        size_t current_peak = peak_memory_usage_.load();
        while (current_usage > current_peak && 
               !peak_memory_usage_.compare_exchange_weak(current_peak, current_usage)) {
            // Retry if compare_exchange_weak failed
        }
    }
    return ptr;
}

void MemoryManager::deallocate(void* ptr) {
    total_deallocations_++;
    global_pool_->deallocate(ptr);
}

void MemoryManager::setGlobalPoolSize(size_t size) {
    global_pool_->reserve(size);
}

void MemoryManager::setThreadLocalPoolSize(size_t size) {
    thread_local_pool_size_ = size;
}

void MemoryManager::printStatistics() const {
    std::cout << "Memory Manager Statistics:" << std::endl;
    std::cout << "  Total Allocations: " << total_allocations_ << std::endl;
    std::cout << "  Total Deallocations: " << total_deallocations_ << std::endl;
    std::cout << "  Peak Memory Usage: " << peak_memory_usage_ << " bytes" << std::endl;
    std::cout << "  Current Allocated: " << global_pool_->getTotalAllocated() << " bytes" << std::endl;
    std::cout << "  Total Reserved: " << global_pool_->getTotalReserved() << " bytes" << std::endl;
    std::cout << "  Fragmentation: " << (global_pool_->getFragmentationRatio() * 100) << "%" << std::endl;
}

void MemoryManager::resetStatistics() {
    total_allocations_ = 0;
    total_deallocations_ = 0;
    peak_memory_usage_ = 0;
}

} // namespace imgreg 
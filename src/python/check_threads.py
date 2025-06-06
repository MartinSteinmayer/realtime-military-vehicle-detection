#!/usr/bin/env python3
import os
import psutil
import multiprocessing
import subprocess

def get_system_info():
    """Get comprehensive system information"""
    
    print("="*60)
    print("SYSTEM RESOURCE INFORMATION")
    print("="*60)
    
    # CPU Information
    print("\n--- CPU Information ---")
    print(f"Physical cores: {psutil.cpu_count(logical=False)}")
    print(f"Logical cores (threads): {psutil.cpu_count(logical=True)}")
    print(f"Multiprocessing cpu_count: {multiprocessing.cpu_count()}")
    print(f"Current CPU usage: {psutil.cpu_percent(interval=1)}%")
    
    # Memory Information
    print("\n--- Memory Information ---")
    mem = psutil.virtual_memory()
    print(f"Total memory: {mem.total / (1024**3):.2f} GB")
    print(f"Available memory: {mem.available / (1024**3):.2f} GB")
    print(f"Used memory: {mem.used / (1024**3):.2f} GB ({mem.percent}%)")
    
    # Process Limits
    print("\n--- Process Limits ---")
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NPROC)
        print(f"Max processes/threads (soft limit): {soft}")
        print(f"Max processes/threads (hard limit): {hard}")
        
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        print(f"Max memory (soft): {soft if soft != -1 else 'unlimited'}")
        print(f"Max memory (hard): {hard if hard != -1 else 'unlimited'}")
    except:
        print("Could not retrieve resource limits")
    
    # Environment Variables
    print("\n--- Relevant Environment Variables ---")
    env_vars = ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
                'SLURM_CPUS_PER_TASK', 'SLURM_JOB_ID', 'PBS_NODEFILE', 
                'SGE_TASK_ID', 'NSLOTS']
    for var in env_vars:
        value = os.environ.get(var, 'Not set')
        if value != 'Not set':
            print(f"{var}: {value}")
    
    # Check if we're in a job scheduler environment
    print("\n--- Job Scheduler Detection ---")
    if 'SLURM_JOB_ID' in os.environ:
        print("SLURM detected!")
        try:
            result = subprocess.run(['scontrol', 'show', 'job', os.environ['SLURM_JOB_ID']], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if 'NumCPUs' in line or 'NumNodes' in line:
                        print(f"  {line.strip()}")
        except:
            pass
            
    elif 'PBS_JOBID' in os.environ:
        print("PBS/Torque detected!")
        
    elif 'SGE_TASK_ID' in os.environ:
        print("SGE detected!")
    else:
        print("No job scheduler detected (or running interactively)")
    
    # Recommendation
    print("\n--- Recommendation for Parallel Processing ---")
    physical_cores = psutil.cpu_count(logical=False)
    logical_cores = psutil.cpu_count(logical=True)
    available_mem_gb = psutil.virtual_memory().available / (1024**3)
    
    # Conservative recommendation
    if logical_cores:
        recommended_workers = min(
            logical_cores // 2,  # Use half of logical cores
            int(available_mem_gb / 2),  # Assume 2GB per worker
            32  # Cap at 32 for safety
        )
        print(f"Recommended worker threads: {recommended_workers}")
        print(f"(Based on {logical_cores} threads and {available_mem_gb:.1f}GB available memory)")
    
    print("\n--- Current Process Info ---")
    process = psutil.Process()
    print(f"Current process threads: {process.num_threads()}")
    print(f"Current process memory: {process.memory_info().rss / (1024**3):.2f} GB")

if __name__ == "__main__":
    get_system_info()
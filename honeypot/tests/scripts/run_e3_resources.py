#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E3 — Resource Profile (CPU/GPU/RAM/VRAM)
- Purpose: Measure operational footprint during steady load
- Metrics:
  * CPU usage (%, mean/peak per core and total)
  * RAM usage (GB, mean/peak)
  * GPU utilization (%, mean/peak)
  * VRAM usage (GB, mean/peak)
  * Power consumption (W) if available
  * Temperature (°C) if available
- Setup: N-minute steady run at target RPS; sample every second
- Outputs:
  * results/resources/resources_raw.csv (per-second samples)
  * results/resources/resources_summary.csv (aggregated stats per model/config)
  * results/resources/system_info.txt (detailed system specs)

Note: Graphical representations generated in post-processing:
  - Fig 3a: Stacked bars (CPU/RAM/GPU/VRAM) per model
  - Fig 3b: Time-series for representative 2-minute window
"""

import argparse
import csv
import os
import sys
import time
import threading
import queue
import platform
import json
from datetime import datetime
from typing import Dict, List, Optional

import psutil
import requests
import pandas as pd
import numpy as np

# Intentar importar bibliotecas de GPU
GPU_AVAILABLE = False
try:
    import pynvml
    pynvml.nvmlInit()
    GPU_AVAILABLE = True
except (ImportError, Exception) as e:
    print(f"[WARN] GPU monitoring not available: {e}")

# -----------------------
# Configuración por defecto
# -----------------------

API_ENDPOINT_API = "http://localhost:5555/webapi/entry.cgi"
API_ENDPOINT_SYS = "http://localhost:80/index.html"

MODELS = ["gemma", "llama3", "zephyr"]

# Request pool para carga sostenida
REQUESTS = [
  "api=SYNO.API.Auth&method=login&version=6&account=admin&passwd=password123",
  "api=SYNO.FileStation.List&method=list_share&version=2",
  "api=SYNO.Core.System&method=info&version=1",
  "api=SYNO.DownloadStation.Task&method=list&version=1",
  "api=SYNO.FileStation.List&method=list&version=2&folder_path=/volume1",
  "api=SYNO.FileStation.Info&method=get&version=2&path=/volume1/public",
  "api=SYNO.Core.User&method=get&version=1&user_name=admin",
  "api=SYNO.Core.System.Utilization&method=get&version=1",
]

OUT_DIR = "results/resources"
RAW_CSV = os.path.join(OUT_DIR, "resources_raw.csv")
SUMMARY_CSV = os.path.join(OUT_DIR, "resources_summary.csv")
SYSTEM_INFO_TXT = os.path.join(OUT_DIR, "system_info.txt")

# -----------------------
# System information collection
# -----------------------

def collect_detailed_system_info() -> dict:
    """
    Recopila información detallada del sistema para el reporte.
    """
    info = {
        "timestamp": datetime.utcnow().isoformat(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "architecture": platform.machine(),
        "hostname": platform.node(),
        "python_version": platform.python_version(),
    }
    
    # CPU info
    info["cpu_count_physical"] = psutil.cpu_count(logical=False)
    info["cpu_count_logical"] = psutil.cpu_count(logical=True)
    cpu_freq = psutil.cpu_freq()
    if cpu_freq:
        info["cpu_freq_current_mhz"] = cpu_freq.current
        info["cpu_freq_min_mhz"] = cpu_freq.min
        info["cpu_freq_max_mhz"] = cpu_freq.max
    
    # RAM info
    mem = psutil.virtual_memory()
    info["ram_total_gb"] = round(mem.total / (1024**3), 2)
    
    # Swap info
    swap = psutil.swap_memory()
    info["swap_total_gb"] = round(swap.total / (1024**3), 2)
    
    # Disk info
    disk = psutil.disk_usage('/')
    info["disk_total_gb"] = round(disk.total / (1024**3), 2)
    info["disk_free_gb"] = round(disk.free / (1024**3), 2)
    
    # GPU info
    if GPU_AVAILABLE:
        try:
            device_count = pynvml.nvmlDeviceGetCount()
            info["gpu_count"] = device_count
            
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                gpu_name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(gpu_name, bytes):
                    gpu_name = gpu_name.decode('utf-8')
                
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                
                info[f"gpu_{i}_name"] = gpu_name
                info[f"gpu_{i}_vram_total_gb"] = round(mem_info.total / (1024**3), 2)
                
                # Driver version
                try:
                    driver_version = pynvml.nvmlSystemGetDriverVersion()
                    if isinstance(driver_version, bytes):
                        driver_version = driver_version.decode('utf-8')
                    info[f"gpu_{i}_driver_version"] = driver_version
                except:
                    pass
        except Exception as e:
            info["gpu_error"] = str(e)
    else:
        info["gpu_available"] = False
    
    return info

def save_system_info(info: dict, filepath: str):
    """
    Guarda información del sistema en archivo de texto.
    """
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("SYSTEM INFORMATION - E3 Resource Profile\n")
        f.write("=" * 70 + "\n\n")
        for key, value in sorted(info.items()):
            f.write(f"{key:30s}: {value}\n")
        f.write("\n" + "=" * 70 + "\n")

# -----------------------
# Resource monitoring
# -----------------------

class ResourceMonitor:
    """
    Monitor de recursos del sistema que muestrea cada segundo.
    """
    def __init__(self, sample_interval: float = 1.0, gpu_id: int = 3):
        self.sample_interval = sample_interval
        self.gpu_id = gpu_id
        self.samples = []
        self.running = False
        self.thread = None
        
    def _sample_resources(self) -> Dict:
        """
        Toma una muestra instantánea de recursos.
        """
        sample = {
            "timestamp": datetime.utcnow().isoformat(),
            "epoch_time": time.time(),
        }
        
        # CPU
        cpu_percent_per_core = psutil.cpu_percent(interval=0, percpu=True)
        sample["cpu_percent_total"] = psutil.cpu_percent(interval=0)
        for i, cpu_pct in enumerate(cpu_percent_per_core):
            sample[f"cpu_percent_core_{i}"] = cpu_pct
        
        # CPU frequency
        cpu_freq = psutil.cpu_freq()
        if cpu_freq:
            sample["cpu_freq_current_mhz"] = cpu_freq.current
        
        # RAM
        mem = psutil.virtual_memory()
        sample["ram_used_gb"] = round(mem.used / (1024**3), 3)
        sample["ram_available_gb"] = round(mem.available / (1024**3), 3)
        sample["ram_percent"] = mem.percent
        
        # Swap
        swap = psutil.swap_memory()
        sample["swap_used_gb"] = round(swap.used / (1024**3), 3)
        sample["swap_percent"] = swap.percent
        
        # GPU
        if GPU_AVAILABLE:
            try:
                device_count = pynvml.nvmlDeviceGetCount()
                # Monitorizar solo la GPU especificada (gpu_id)
                if self.gpu_id < device_count:
                    i = self.gpu_id
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    
                    # Utilization
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    sample[f"gpu_{i}_util_percent"] = util.gpu
                    sample[f"gpu_{i}_mem_util_percent"] = util.memory
                    
                    # Memory
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    sample[f"gpu_{i}_vram_used_gb"] = round(mem_info.used / (1024**3), 3)
                    sample[f"gpu_{i}_vram_free_gb"] = round(mem_info.free / (1024**3), 3)
                    sample[f"gpu_{i}_vram_total_gb"] = round(mem_info.total / (1024**3), 3)
                    
                    # Temperature
                    try:
                        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                        sample[f"gpu_{i}_temp_c"] = temp
                    except:
                        pass
                    
                    # Power
                    try:
                        power = pynvml.nvmlDeviceGetPowerUsage(handle)  # mW
                        sample[f"gpu_{i}_power_w"] = round(power / 1000.0, 2)
                    except:
                        pass
                    
                    # Clock speeds
                    try:
                        clock_sm = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
                        clock_mem = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
                        sample[f"gpu_{i}_clock_sm_mhz"] = clock_sm
                        sample[f"gpu_{i}_clock_mem_mhz"] = clock_mem
                    except:
                        pass
            except Exception as e:
                sample["gpu_error"] = str(e)
        
        return sample
    
    def _monitor_loop(self):
        """
        Loop de monitoreo que corre en thread separado.
        """
        while self.running:
            try:
                sample = self._sample_resources()
                self.samples.append(sample)
            except Exception as e:
                print(f"[WARN] Error sampling resources: {e}")
            
            time.sleep(self.sample_interval)
    
    def start(self):
        """
        Inicia el monitoreo en background.
        """
        if self.running:
            return
        
        self.running = True
        self.samples = []
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        print(f"[INFO] Resource monitoring started (sampling every {self.sample_interval}s)")
    
    def stop(self):
        """
        Detiene el monitoreo.
        """
        if not self.running:
            return
        
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        print(f"[INFO] Resource monitoring stopped. Collected {len(self.samples)} samples.")
    
    def get_samples(self) -> List[Dict]:
        """
        Retorna todas las muestras recopiladas.
        """
        return self.samples.copy()

# -----------------------
# Load generator
# -----------------------

class LoadGenerator:
    """
    Generador de carga que mantiene un RPS objetivo constante.
    """
    def __init__(self, target_rps: float, base_url: str, model: str, 
                 requests_pool: List[str], timeout: float = 60.0):
        self.target_rps = target_rps
        self.base_url = base_url
        self.model = model
        self.requests_pool = requests_pool
        self.timeout = timeout
        self.running = False
        self.threads = []
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_latency_ms": 0.0
        }
        self.stats_lock = threading.Lock()
        
    def _worker(self, worker_id: int, interval: float):
        """
        Worker que envía requests a intervalos regulares.
        """
        request_num = 0
        while self.running:
            request_num += 1
            req_query = self.requests_pool[request_num % len(self.requests_pool)]
            url = f"{self.base_url}?{req_query}&model_name={self.model}"
            
            t_start = time.perf_counter()
            try:
                resp = requests.get(url, timeout=self.timeout)
                t_end = time.perf_counter()
                latency_ms = (t_end - t_start) * 1000
                
                with self.stats_lock:
                    self.stats["total_requests"] += 1
                    if resp.status_code == 200:
                        self.stats["successful_requests"] += 1
                    else:
                        self.stats["failed_requests"] += 1
                    self.stats["total_latency_ms"] += latency_ms
            except Exception:
                with self.stats_lock:
                    self.stats["total_requests"] += 1
                    self.stats["failed_requests"] += 1
            
            time.sleep(interval)
    
    def start(self, num_workers: int = 1):
        """
        Inicia la generación de carga.
        num_workers: número de threads concurrentes
        """
        if self.running:
            return
        
        self.running = True
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_latency_ms": 0.0
        }
        
        # Calcular intervalo entre requests por worker
        interval = num_workers / self.target_rps if self.target_rps > 0 else 1.0
        
        for i in range(num_workers):
            thread = threading.Thread(
                target=self._worker, 
                args=(i, interval),
                daemon=True
            )
            thread.start()
            self.threads.append(thread)
        
        print(f"[INFO] Load generator started: target {self.target_rps} RPS with {num_workers} workers")
    
    def stop(self):
        """
        Detiene la generación de carga.
        """
        if not self.running:
            return
        
        self.running = False
        for thread in self.threads:
            thread.join(timeout=2)
        self.threads = []
        
        with self.stats_lock:
            stats_copy = self.stats.copy()
        
        print(f"[INFO] Load generator stopped.")
        print(f"  Total requests: {stats_copy['total_requests']}")
        print(f"  Successful: {stats_copy['successful_requests']}")
        print(f"  Failed: {stats_copy['failed_requests']}")
        if stats_copy['total_requests'] > 0:
            avg_latency = stats_copy['total_latency_ms'] / stats_copy['total_requests']
            print(f"  Avg latency: {avg_latency:.2f}ms")
    
    def get_stats(self) -> Dict:
        """
        Retorna estadísticas actuales.
        """
        with self.stats_lock:
            return self.stats.copy()

# -----------------------
# Summary statistics
# -----------------------

def calculate_summary_stats(samples: List[Dict], load_stats: Dict, 
                           test_config: Dict, gpu_id: int = 3) -> Dict:
    """
    Calcula estadísticas agregadas de las muestras de recursos.
    """
    if not samples:
        return {}
    
    df = pd.DataFrame(samples)
    
    gpu_col_prefix = f"gpu_{gpu_id}_"
    
    summary = {
        "model": test_config.get("model"),
        "pipeline": test_config.get("pipeline"),
        "duration_sec": test_config.get("duration_sec"),
        "target_rps": test_config.get("target_rps"),
        "num_samples": len(samples),
    }
    
    # CPU stats
    if "cpu_percent_total" in df.columns:
        summary["cpu_percent_mean"] = df["cpu_percent_total"].mean()
        summary["cpu_percent_peak"] = df["cpu_percent_total"].max()
        summary["cpu_percent_std"] = df["cpu_percent_total"].std()
    
    # RAM stats
    if "ram_used_gb" in df.columns:
        summary["ram_used_gb_mean"] = df["ram_used_gb"].mean()
        summary["ram_used_gb_peak"] = df["ram_used_gb"].max()
        summary["ram_percent_mean"] = df["ram_percent"].mean()
        summary["ram_percent_peak"] = df["ram_percent"].max()
    
    # GPU stats (usar GPU especificada)
    if f"{gpu_col_prefix}util_percent" in df.columns:
        summary["gpu_util_percent_mean"] = df[f"{gpu_col_prefix}util_percent"].mean()
        summary["gpu_util_percent_peak"] = df[f"{gpu_col_prefix}util_percent"].max()
    
    if f"{gpu_col_prefix}vram_used_gb" in df.columns:
        summary["vram_used_gb_mean"] = df[f"{gpu_col_prefix}vram_used_gb"].mean()
        summary["vram_used_gb_peak"] = df[f"{gpu_col_prefix}vram_used_gb"].max()
    
    if f"{gpu_col_prefix}temp_c" in df.columns:
        summary["gpu_temp_c_mean"] = df[f"{gpu_col_prefix}temp_c"].mean()
        summary["gpu_temp_c_peak"] = df[f"{gpu_col_prefix}temp_c"].max()
    
    if f"{gpu_col_prefix}power_w" in df.columns:
        summary["gpu_power_w_mean"] = df[f"{gpu_col_prefix}power_w"].mean()
        summary["gpu_power_w_peak"] = df[f"{gpu_col_prefix}power_w"].max()
    
    # Load generator stats
    summary["total_requests"] = load_stats.get("total_requests", 0)
    summary["successful_requests"] = load_stats.get("successful_requests", 0)
    summary["failed_requests"] = load_stats.get("failed_requests", 0)
    
    if summary["total_requests"] > 0:
        summary["success_rate"] = summary["successful_requests"] / summary["total_requests"]
        summary["actual_rps"] = summary["total_requests"] / test_config.get("duration_sec", 1)
        if load_stats.get("total_latency_ms", 0) > 0:
            summary["avg_latency_ms"] = load_stats["total_latency_ms"] / summary["total_requests"]
    
    return summary

# -----------------------
# Main experiment
# -----------------------

def main():
    parser = argparse.ArgumentParser(
        description="E3 Resource Profile: Measure CPU/GPU/RAM/VRAM during steady load"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=600.0,
        help="Test duration in seconds (default: 600s = 10 minutes)"
    )
    parser.add_argument(
        "--target-rps",
        type=float,
        default=10.0,
        help="Target requests per second to maintain (default: 10)"
    )
    parser.add_argument(
        "--sample-interval",
        type=float,
        default=1.0,
        help="Resource sampling interval in seconds (default: 1.0)"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Number of concurrent worker threads for load generation (default: 1)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemma",
        choices=MODELS,
        help="Model to test"
    )
    parser.add_argument(
        "--pipeline",
        type=str,
        default="system",
        choices=["api", "system"],
        help="Pipeline: 'api' (direct) or 'system' (full stack)"
    )
    parser.add_argument(
        "--warmup",
        type=float,
        default=5.0,
        help="Warmup duration in seconds before starting measurement (default: 5)"
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=3,
        help="GPU device ID to monitor (default: 3)"
    )
    
    args = parser.parse_args()
    
    # Select endpoint
    base_url = API_ENDPOINT_API if args.pipeline == "api" else API_ENDPOINT_SYS
    
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # Recopilar información del sistema
    print("\n[INFO] Collecting system information...")
    sys_info = collect_detailed_system_info()
    save_system_info(sys_info, SYSTEM_INFO_TXT)
    print(f"[INFO] System info saved to: {SYSTEM_INFO_TXT}")
    
    print(f"\n{'='*70}")
    print(f"E3 RESOURCE PROFILE - {args.model.upper()} - {args.pipeline.upper()}")
    print(f"{'='*70}")
    print(f"Model: {args.model}")
    print(f"Pipeline: {args.pipeline}")
    print(f"Duration: {args.duration}s")
    print(f"Target RPS: {args.target_rps}")
    print(f"Sample interval: {args.sample_interval}s")
    print(f"Workers: {args.num_workers}")
    print(f"Warmup: {args.warmup}s")
    print(f"Endpoint: {base_url}")
    print(f"{'='*70}\n")
    
    # Configuración del test
    test_config = {
        "model": args.model,
        "pipeline": args.pipeline,
        "duration_sec": args.duration,
        "target_rps": args.target_rps,
        "sample_interval": args.sample_interval,
        "num_workers": args.num_workers,
        "warmup_sec": args.warmup
    }
    
    # Inicializar componentes
    monitor = ResourceMonitor(sample_interval=args.sample_interval, gpu_id=args.gpu_id)
    load_gen = LoadGenerator(
        target_rps=args.target_rps,
        base_url=base_url,
        model=args.model,
        requests_pool=REQUESTS
    )
    
    # Warmup
    if args.warmup > 0:
        print(f"[⚡] Warmup phase ({args.warmup}s)...")
        load_gen.start(num_workers=args.num_workers)
        time.sleep(args.warmup)
        load_gen.stop()
        print("[✓] Warmup completed\n")
        time.sleep(2)  # Pequeña pausa
    
    # Iniciar monitoreo
    print(f"[▶] Starting resource monitoring and load generation...")
    monitor.start()
    load_gen.start(num_workers=args.num_workers)
    
    # Ejecutar durante la duración especificada
    start_time = time.time()
    try:
        while time.time() - start_time < args.duration:
            elapsed = time.time() - start_time
            remaining = args.duration - elapsed
            
            # Mostrar progreso cada 30 segundos
            if int(elapsed) % 30 == 0 and int(elapsed) > 0:
                stats = load_gen.get_stats()
                current_rps = stats["total_requests"] / elapsed if elapsed > 0 else 0
                print(f"[{int(elapsed)}s] Progress: {elapsed/args.duration*100:.1f}% | "
                      f"RPS: {current_rps:.2f} | "
                      f"Requests: {stats['total_requests']} | "
                      f"Failures: {stats['failed_requests']}")
            
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[!] Interrupted by user")
    
    # Detener todo
    print("\n[■] Stopping monitoring and load generation...")
    load_gen.stop()
    monitor.stop()
    
    # Recopilar datos
    samples = monitor.get_samples()
    load_stats = load_gen.get_stats()
    
    print(f"\n[INFO] Processing {len(samples)} resource samples...")
    
    # Guardar datos raw
    if samples:
        df_raw = pd.DataFrame(samples)
        
        # Añadir metadata del test
        df_raw.insert(0, "test_model", args.model)
        df_raw.insert(1, "test_pipeline", args.pipeline)
        
        df_raw.to_csv(RAW_CSV, index=False)
        print(f"[✓] Raw data saved: {RAW_CSV}")
    
    # Calcular y guardar resumen
    summary = calculate_summary_stats(samples, load_stats, test_config, gpu_id=args.gpu_id)
    
    # Añadir metadata adicional
    summary["test_timestamp"] = datetime.utcnow().isoformat()
    
    df_summary = pd.DataFrame([summary])
    df_summary.to_csv(SUMMARY_CSV, index=False)
    print(f"[✓] Summary saved: {SUMMARY_CSV}")
    
    # Mostrar resumen
    print(f"\n{'='*70}")
    print("✅ RESOURCE PROFILING COMPLETED")
    print(f"{'='*70}")
    print(f"\n📊 RESOURCE USAGE SUMMARY:")
    print(f"  CPU (mean/peak):      {summary.get('cpu_percent_mean', 0):.1f}% / {summary.get('cpu_percent_peak', 0):.1f}%")
    print(f"  RAM (mean/peak):      {summary.get('ram_used_gb_mean', 0):.2f} GB / {summary.get('ram_used_gb_peak', 0):.2f} GB")
    
    if "gpu_util_percent_mean" in summary:
        print(f"  GPU Util (mean/peak): {summary.get('gpu_util_percent_mean', 0):.1f}% / {summary.get('gpu_util_percent_peak', 0):.1f}%")
        print(f"  VRAM (mean/peak):     {summary.get('vram_used_gb_mean', 0):.2f} GB / {summary.get('vram_used_gb_peak', 0):.2f} GB")
        
        if "gpu_temp_c_mean" in summary:
            print(f"  GPU Temp (mean/peak): {summary.get('gpu_temp_c_mean', 0):.1f}°C / {summary.get('gpu_temp_c_peak', 0):.1f}°C")
        
        if "gpu_power_w_mean" in summary:
            print(f"  GPU Power (mean/peak): {summary.get('gpu_power_w_mean', 0):.1f}W / {summary.get('gpu_power_w_peak', 0):.1f}W")
    
    print(f"\n📊 LOAD GENERATOR SUMMARY:")
    print(f"  Target RPS:           {args.target_rps:.2f}")
    print(f"  Actual RPS:           {summary.get('actual_rps', 0):.2f}")
    print(f"  Total requests:       {summary.get('total_requests', 0)}")
    print(f"  Success rate:         {summary.get('success_rate', 0)*100:.2f}%")
    if "avg_latency_ms" in summary:
        print(f"  Avg latency:          {summary.get('avg_latency_ms', 0):.2f}ms")
    
    print(f"\n📁 Output files:")
    print(f"  • {RAW_CSV}")
    print(f"  • {SUMMARY_CSV}")
    print(f"  • {SYSTEM_INFO_TXT}")
    
    print(f"\n📊 Generate graphs with:")
    print(f"  python plot_e3_graphs.py --summary-csv {SUMMARY_CSV}")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()

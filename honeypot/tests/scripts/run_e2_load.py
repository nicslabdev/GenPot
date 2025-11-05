#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E2 — Throughput & Scalability under Load
- Purpose: Measure RPS (requests per second) capacity before SLO breaches
- Pipelines: API-only (http://localhost:5555) vs System (http://localhost:80)
- Metrics: 
  * RPS at p95 ≤ X ms threshold
  * Error rate (% timeouts/failures)
  * Queueing delay
  * Throughput vs latency (knee curve)
- Outputs:
  * results/load/load_raw.csv (per-request timings)
  * results/load/load_summary.csv (aggregated by concurrency level)
  * results/load/hardware_info.txt (system specs for reporting)

Note: Uses closed-loop load testing with concurrent sessions (1→N).
Graphical representations generated in post-processing.
"""

import argparse
import csv
import os
import sys
import time
import json
import threading
import queue
import platform
import psutil
from datetime import datetime
from collections import defaultdict
from typing import List, Dict, Tuple

import requests
import pandas as pd
import numpy as np

# -----------------------
# Configuración por defecto
# -----------------------

API_ENDPOINT_API = "http://localhost:5555/webapi/entry.cgi"   # API-only
API_ENDPOINT_SYS = "http://localhost:80/index.html"     # Full system

MODELS = ["gemma", "llama3", "zephyr"]

# Conjunto de requests para replay (muestreo bajo carga)
REQUESTS = [
  "api=SYNO.API.Auth&method=login&version=6&account=admin&passwd=password123",
  "api=SYNO.FileStation.List&method=list_share&version=2",
  "api=SYNO.Core.System&method=info&version=1",
  "api=SYNO.DownloadStation.Task&method=list&version=1",
  "api=SYNO.FileStation.List&method=list&version=2&folder_path=/volume1",
  "api=SYNO.FileStation.Info&method=get&version=2&path=/volume1/public",
  "api=SYNO.Core.User&method=get&version=1&user_name=admin",
  "api=SYNO.Core.User&method=logout&version=1&user_name=admin",
  "api=SYNO.Core.System.Utilization&method=get&version=1",
  "api=SYNO.Core.System.Status&method=network_status&version=1",
]

OUT_DIR = "results/load"
RAW_CSV = os.path.join(OUT_DIR, "load_raw.csv")
SUMMARY_CSV = os.path.join(OUT_DIR, "load_summary.csv")
HARDWARE_TXT = os.path.join(OUT_DIR, "hardware_info.txt")

# -----------------------
# Hardware information
# -----------------------

def collect_hardware_info() -> dict:
    """
    Recopila información del hardware para el reporte.
    """
    info = {
        "timestamp": datetime.utcnow().isoformat(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": psutil.cpu_count(logical=False),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_freq_mhz": psutil.cpu_freq().current if psutil.cpu_freq() else "N/A",
        "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "ram_available_gb": round(psutil.virtual_memory().available / (1024**3), 2),
        "python_version": platform.python_version(),
    }
    
    # Información de GPU (si está disponible)
    try:
        import torch
        if torch.cuda.is_available():
            info["gpu_available"] = True
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_count"] = torch.cuda.device_count()
            info["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
        else:
            info["gpu_available"] = False
    except ImportError:
        info["gpu_available"] = "torch not available"
    
    return info

def save_hardware_info(info: dict, filepath: str):
    """
    Guarda la información del hardware en un archivo de texto.
    """
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("HARDWARE INFORMATION - E2 Load Test\n")
        f.write("=" * 60 + "\n\n")
        for key, value in info.items():
            f.write(f"{key:25s}: {value}\n")
        f.write("\n" + "=" * 60 + "\n")

# -----------------------
# Load testing worker
# -----------------------

class LoadWorker(threading.Thread):
    """
    Worker thread que ejecuta requests continuamente durante un período de tiempo.
    """
    def __init__(self, worker_id: int, base_url: str, model: str, 
                 requests_pool: List[str], duration_sec: float, 
                 timeout: float, results_queue: queue.Queue):
        super().__init__()
        self.worker_id = worker_id
        self.base_url = base_url
        self.model = model
        self.requests_pool = requests_pool
        self.duration_sec = duration_sec
        self.timeout = timeout
        self.results_queue = results_queue
        self.stop_flag = threading.Event()
        
    def run(self):
        """
        Ejecuta requests en loop hasta que se cumpla la duración.
        """
        start_time = time.time()
        request_count = 0
        
        while not self.stop_flag.is_set():
            elapsed = time.time() - start_time
            if elapsed >= self.duration_sec:
                break
            
            # Seleccionar request de forma cíclica
            req_query = self.requests_pool[request_count % len(self.requests_pool)]
            request_count += 1
            
            # Construir URL
            url = f"{self.base_url}?{req_query}&model_name={self.model}"
            
            # Medir tiempo de respuesta
            t_start = time.perf_counter()
            t_start_ns = time.perf_counter_ns()
            
            try:
                resp = requests.get(url, timeout=self.timeout)
                t_end_ns = time.perf_counter_ns()
                
                latency_ms = (t_end_ns - t_start_ns) / 1e6
                
                result = {
                    "worker_id": self.worker_id,
                    "timestamp": datetime.utcnow().isoformat(),
                    "request_num": request_count,
                    "status": resp.status_code,
                    "latency_ms": latency_ms,
                    "bytes": len(resp.content),
                    "error": 0,
                    "timeout": 0
                }
            except requests.exceptions.Timeout:
                t_end_ns = time.perf_counter_ns()
                latency_ms = (t_end_ns - t_start_ns) / 1e6
                result = {
                    "worker_id": self.worker_id,
                    "timestamp": datetime.utcnow().isoformat(),
                    "request_num": request_count,
                    "status": None,
                    "latency_ms": latency_ms,
                    "bytes": 0,
                    "error": 1,
                    "timeout": 1
                }
            except Exception as e:
                t_end_ns = time.perf_counter_ns()
                latency_ms = (t_end_ns - t_start_ns) / 1e6
                result = {
                    "worker_id": self.worker_id,
                    "timestamp": datetime.utcnow().isoformat(),
                    "request_num": request_count,
                    "status": None,
                    "latency_ms": latency_ms,
                    "bytes": 0,
                    "error": 1,
                    "timeout": 0
                }
            
            self.results_queue.put(result)
    
    def stop(self):
        """
        Señal para detener el worker.
        """
        self.stop_flag.set()

# -----------------------
# Load test execution
# -----------------------

def run_load_test(concurrency: int, base_url: str, model: str, 
                  requests_pool: List[str], duration_sec: float, 
                  timeout: float) -> List[dict]:
    """
    Ejecuta un test de carga con N workers concurrentes durante duration_sec.
    Retorna lista de resultados individuales.
    """
    results_queue = queue.Queue()
    workers = []
    
    print(f"  [→] Starting {concurrency} concurrent workers for {duration_sec}s...")
    
    # Crear y arrancar workers
    for i in range(concurrency):
        worker = LoadWorker(
            worker_id=i,
            base_url=base_url,
            model=model,
            requests_pool=requests_pool,
            duration_sec=duration_sec,
            timeout=timeout,
            results_queue=results_queue
        )
        workers.append(worker)
        worker.start()
    
    # Esperar a que todos terminen
    for worker in workers:
        worker.join()
    
    # Recolectar resultados
    results = []
    while not results_queue.empty():
        results.append(results_queue.get())
    
    print(f"  [✓] Completed. Total requests: {len(results)}")
    
    return results

# -----------------------
# Metrics calculation
# -----------------------

def calculate_metrics(results: List[dict], duration_sec: float) -> dict:
    """
    Calcula métricas agregadas de un test de carga:
    - RPS (requests per second)
    - Error rate
    - Latency percentiles (p50, p95, p99)
    - Queueing delay estimate
    """
    if not results:
        return {
            "total_requests": 0,
            "rps": 0.0,
            "error_rate": 0.0,
            "timeout_rate": 0.0,
            "latency_p50_ms": None,
            "latency_p95_ms": None,
            "latency_p99_ms": None,
            "latency_mean_ms": None,
            "latency_std_ms": None,
            "queueing_delay_ms": None
        }
    
    total = len(results)
    errors = sum(1 for r in results if r["error"] == 1)
    timeouts = sum(1 for r in results if r["timeout"] == 1)
    
    # Filtrar solo requests exitosos para latencia
    successful = [r for r in results if r["error"] == 0]
    latencies = [r["latency_ms"] for r in successful]
    
    if latencies:
        p50 = np.percentile(latencies, 50)
        p95 = np.percentile(latencies, 95)
        p99 = np.percentile(latencies, 99)
        mean = np.mean(latencies)
        std = np.std(latencies)
        
        # Queueing delay: diferencia entre p95 y p50 como proxy
        queueing = p95 - p50
    else:
        p50 = p95 = p99 = mean = std = queueing = None
    
    rps = total / duration_sec if duration_sec > 0 else 0.0
    
    return {
        "total_requests": total,
        "rps": rps,
        "error_rate": errors / total if total > 0 else 0.0,
        "timeout_rate": timeouts / total if total > 0 else 0.0,
        "latency_p50_ms": p50,
        "latency_p95_ms": p95,
        "latency_p99_ms": p99,
        "latency_mean_ms": mean,
        "latency_std_ms": std,
        "queueing_delay_ms": queueing
    }

# -----------------------
# Main experiment
# -----------------------

def main():
    parser = argparse.ArgumentParser(
        description="E2 Load Testing: Throughput & Scalability under Load"
    )
    parser.add_argument(
        "--concurrency-levels", 
        type=str, 
        default="1,2,4,8,16,32,64",
        help="Comma-separated concurrency levels to test (e.g., '1,2,4,8,16')"
    )
    parser.add_argument(
        "--duration-per-level", 
        type=float, 
        default=30.0,
        help="Duration (seconds) for each concurrency level"
    )
    parser.add_argument(
        "--warmup-duration", 
        type=float, 
        default=5.0,
        help="Warmup duration (seconds) before each level"
    )
    parser.add_argument(
        "--timeout", 
        type=float, 
        default=60.0,
        help="HTTP request timeout (seconds)"
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
        default="api",
        choices=["api", "system"],
        help="Pipeline to test: 'api' (direct) or 'system' (full stack)"
    )
    parser.add_argument(
        "--cooldown", 
        type=float, 
        default=10.0,
        help="Cooldown time (seconds) between concurrency levels"
    )
    
    args = parser.parse_args()
    
    # Parse concurrency levels
    try:
        concurrency_levels = [int(x.strip()) for x in args.concurrency_levels.split(",")]
        concurrency_levels = sorted(concurrency_levels)
    except ValueError:
        print("[ERROR] Invalid concurrency-levels format. Use comma-separated integers.")
        sys.exit(1)
    
    # Select endpoint
    base_url = API_ENDPOINT_API if args.pipeline == "api" else API_ENDPOINT_SYS
    
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # Recopilar información del hardware
    print("\n[INFO] Collecting hardware information...")
    hw_info = collect_hardware_info()
    save_hardware_info(hw_info, HARDWARE_TXT)
    print(f"[INFO] Hardware info saved to: {HARDWARE_TXT}")
    
    # Preparar CSV para resultados raw
    fieldnames_raw = [
        "concurrency", "worker_id", "timestamp", "request_num", 
        "status", "latency_ms", "bytes", "error", "timeout",
        "model", "pipeline"
    ]
    
    with open(RAW_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames_raw)
        writer.writeheader()
    
    print(f"\n{'='*60}")
    print(f"E2 LOAD TEST - {args.model.upper()} - {args.pipeline.upper()}")
    print(f"{'='*60}")
    print(f"Concurrency levels: {concurrency_levels}")
    print(f"Duration per level: {args.duration_per_level}s")
    print(f"Warmup: {args.warmup_duration}s")
    print(f"Cooldown: {args.cooldown}s")
    print(f"Timeout: {args.timeout}s")
    print(f"Endpoint: {base_url}")
    print(f"{'='*60}\n")
    
    summary_results = []
    
    for concurrency in concurrency_levels:
        print(f"\n[CONCURRENCY={concurrency}]")
        
        # Warmup
        if args.warmup_duration > 0:
            print(f"  [⚡] Warmup ({args.warmup_duration}s)...")
            _ = run_load_test(
                concurrency=concurrency,
                base_url=base_url,
                model=args.model,
                requests_pool=REQUESTS,
                duration_sec=args.warmup_duration,
                timeout=args.timeout
            )
        
        # Test real
        results = run_load_test(
            concurrency=concurrency,
            base_url=base_url,
            model=args.model,
            requests_pool=REQUESTS,
            duration_sec=args.duration_per_level,
            timeout=args.timeout
        )
        
        # Guardar resultados raw
        with open(RAW_CSV, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames_raw)
            for r in results:
                row = {
                    "concurrency": concurrency,
                    "model": args.model,
                    "pipeline": args.pipeline,
                    **r
                }
                writer.writerow(row)
        
        # Calcular métricas
        metrics = calculate_metrics(results, args.duration_per_level)
        
        summary_row = {
            "concurrency": concurrency,
            "model": args.model,
            "pipeline": args.pipeline,
            "duration_sec": args.duration_per_level,
            **metrics
        }
        summary_results.append(summary_row)
        
        # Mostrar resumen
        print(f"  [📊] RPS: {metrics['rps']:.2f}")
        print(f"  [📊] Error rate: {metrics['error_rate']*100:.2f}%")
        
        # Solo mostrar latencias si hay datos válidos
        if metrics['latency_p50_ms'] is not None:
            print(f"  [📊] p50: {metrics['latency_p50_ms']:.1f}ms | p95: {metrics['latency_p95_ms']:.1f}ms | p99: {metrics['latency_p99_ms']:.1f}ms")
            print(f"  [📊] Queueing delay: {metrics['queueing_delay_ms']:.1f}ms")
        else:
            print(f"  [⚠️] No successful requests - latency data unavailable")
        
        # Cooldown
        if args.cooldown > 0 and concurrency != concurrency_levels[-1]:
            print(f"  [💤] Cooldown ({args.cooldown}s)...")
            time.sleep(args.cooldown)
    
    # Guardar resumen
    df_summary = pd.DataFrame(summary_results)
    df_summary.to_csv(SUMMARY_CSV, index=False)
    
    print(f"\n{'='*60}")
    print("✅ LOAD TEST COMPLETED")
    print(f"{'='*60}")
    print(f"• Raw results: {RAW_CSV}")
    print(f"• Summary: {SUMMARY_CSV}")
    print(f"• Hardware info: {HARDWARE_TXT}")
    print(f"\n📊 Note: Generate graphs using the CSV files:")
    print(f"   - Fig 2a: RPS vs p95 latency (knee curve)")
    print(f"   - Fig 2b: Concurrency vs error rate")
    print(f"{'='*60}\n")
    
    # Mostrar tabla resumen
    print("\n=== SUMMARY TABLE ===")
    print(df_summary[[
        "concurrency", "rps", "error_rate", "latency_p50_ms", 
        "latency_p95_ms", "latency_p99_ms", "queueing_delay_ms"
    ]].round(2).to_string(index=False))
    print()

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E1 — End-to-end responsiveness (latency) & Correctness evaluation
- Pipelines: API-only (http://localhost:5555) vs System (http://localhost:80)
- Networks: LAN vs WAN simulated (tc/netem on loopback)
- Metrics: 
  * Latency: TTFB, TTLB, jitter (std & IQR), timeout rate
  * Correctness: parse rate, accuracy (success/error match), similarity (Levenshtein), key coverage
- Outputs:
  * results/latency/latency_raw.csv (per-request measurements with correctness)
  * results/latency/summary_percentiles.csv (aggregated latency stats with jitter)
  * results/latency/correctness_summary.csv (aggregated correctness metrics)
  
Note: Graphical representations are generated in post-processing from CSV data.
"""

import argparse
import csv
import os
import sys
import time
import socket
import subprocess
from datetime import datetime
from urllib.parse import urlparse, urlunparse, urlencode

import requests
import pandas as pd
import numpy as np
import json
from Levenshtein import distance as lev

# -----------------------
# Configuración por defecto
# -----------------------

API_ENDPOINT_API = "http://localhost:5555/webapi/entry.cgi"   # API-only
API_ENDPOINT_SYS = "http://localhost:80/index.html"     # Full system (docker/reverse proxy)

MODELS = ["gemma", "llama3", "zephyr"]

REQUESTS = [
  "api=SYNO.FileStation.List&method=list_share&version=2",
  "api=SYNO.Core.System&method=info&version=1",
  "api=SYNO.DownloadStation.Task&method=list&version=1",
  "api=SYNO.FileStation.List&method=list&version=2&folder_path=/volume1",
  "api=SYNO.FileStation.Info&method=get&version=2&path=/volume1/public",
  "api=SYNO.Core.User&method=get&version=1&user_name=admin",
  "api=SYNO.Core.User&method=logout&version=1&user_name=admin",
  "api=SYNO.Core.System.Utilization&method=get&version=1",
  "api=SYNO.Core.System.Status&method=network_status&version=1",
  "api=SYNO.API.Auth&method=login&version=6&account=admin&passwd=1234",
  "api=SYNO.API.Auth&method=logout&version=6&session=DownloadStation",
  "api=SYNO.DownloadStation.Task&method=create&version=1&uri=http://example.com/file.iso",
  "api=SYNO.DownloadStation.Task&method=delete&version=1&id=taskid_123",
  "api=SYNO.Core.Network&method=list&version=1",
  "api=SYNO.Core.Storage.Volume&method=status&version=1",
  "api=SYNO.Core.ExternalDevice&method=list&version=1",
  "api=SYNO.Core.Time&method=get&version=1",
  "api=SYNO.FakeModule.Bogus&method=nonexistent&version=1",
  "api=SYNO.Core.System&method=info&version=999",
  "api=SYNO.Core.User&method=delete&version=1&user_name=",
  "api=SYNO.Core.User&metod=login&version=1"  # typo a propósito
]

OUT_DIR = "results/latency"
RAW_CSV = os.path.join(OUT_DIR, "latency_raw.csv")
SUMMARY_CSV = os.path.join(OUT_DIR, "summary_percentiles.csv")
CORRECTNESS_CSV = os.path.join(OUT_DIR, "correctness_summary.csv")
GOLD_FILE = "../fine_tuning/utils/test/gold.json"  # Ruta relativa desde honeypot/fastapi

# -----------------------
# WAN simulation helpers (tc/netem on loopback)
# -----------------------

def tc_apply(profile_name: str, delay_ms: int = 80, loss_pct: float = 0.5) -> bool:
    """
    Apply netem on loopback (lo). Requires sudo/root. Returns True if applied.
    """
    try:
        # Clean existing first
        subprocess.run(["sudo", "tc", "qdisc", "del", "dev", "lo", "root"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # Apply netem
        cmd = ["sudo", "tc", "qdisc", "add", "dev", "lo", "root", "netem", "delay", f"{delay_ms}ms", "loss", f"{loss_pct}%"]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print(f"[WARN] Could not apply netem ({profile_name}): {res.stderr.strip()}")
            return False
        print(f"[INFO] WAN profile '{profile_name}' applied on loopback: delay={delay_ms}ms, loss={loss_pct}%")
        return True
    except FileNotFoundError:
        print("[WARN] 'tc' not found; skipping WAN simulation.")
        return False
    except Exception as e:
        print(f"[WARN] netem exception: {e}")
        return False

def tc_clear():
    try:
        subprocess.run(["sudo", "tc", "qdisc", "del", "dev", "lo", "root"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("[INFO] Cleared netem.")
    except Exception:
        pass

# -----------------------
# Gold reference loading
# -----------------------

def load_gold_responses(gold_path: str) -> dict:
    """
    Carga el archivo gold.json y crea un mapping de request_slug -> expected response.
    El archivo gold puede ser una lista de objetos con 'request' y 'expected'.
    """
    if not os.path.exists(gold_path):
        print(f"[WARN] Gold file not found: {gold_path}")
        return {}
    
    try:
        with open(gold_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        
        gold_map = {}
        if isinstance(raw, list):
            for item in raw:
                if "request" in item and "expected" in item:
                    # Crear slug de la misma forma que en el experimento
                    req = item["request"]
                    slug = "".join([c if c.isalnum() else "_" for c in req])[:80]
                    gold_map[slug] = item["expected"]
        elif isinstance(raw, dict):
            # Si ya es un dict slug->expected
            gold_map = raw
        
        print(f"[INFO] Loaded {len(gold_map)} gold responses from {gold_path}")
        return gold_map
    except Exception as e:
        print(f"[WARN] Error loading gold file: {e}")
        return {}

# -----------------------
# Correctness evaluation
# -----------------------

def evaluate_response(response_text: str, gold_obj: dict) -> dict:
    """
    Evalúa una respuesta contra el gold standard.
    Retorna métricas: ok_parse, class_ok, dist_raw, similarity, key_match
    """
    # Intentar parsear la respuesta
    try:
        resp_obj = json.loads(response_text)
        ok_parse = True
    except Exception:
        resp_obj = {}
        ok_parse = False
    
    # Verificar si success coincide
    class_ok = ok_parse and (resp_obj.get("success") == gold_obj.get("success"))
    
    # Levenshtein sobre strings canónicas
    a = json.dumps(resp_obj, sort_keys=True, separators=(",", ":"))
    b = json.dumps(gold_obj, sort_keys=True, separators=(",", ":"))
    d_raw = lev(a, b)
    L = max(len(a), len(b)) or 1
    similarity = 1 - d_raw / L
    
    # Cobertura de campos (key matching)
    if ok_parse and isinstance(gold_obj, dict) and isinstance(resp_obj, dict):
        keys_r = set(resp_obj.keys())
        keys_g = set(gold_obj.keys())
        key_match = len(keys_r & keys_g) / len(keys_g) if keys_g else 0.0
    else:
        key_match = 0.0 if not ok_parse else None
    
    return {
        "ok_parse": 1 if ok_parse else 0,
        "class_ok": 1 if class_ok else 0,
        "dist_raw": d_raw,
        "similarity": similarity,
        "key_match": key_match if key_match is not None else 0.0
    }

# -----------------------
# Medición TTFB/TTLB
# -----------------------

def measure_http_stream(url: str, timeout: float = 60.0) -> dict:
    """
    Mide TTFB/TTLB en una petición GET con stream=True.
    - TTFB: desde antes de enviar hasta recibir el primer byte.
    - TTLB: hasta recibir todos los bytes.
    También devuelve el contenido completo de la respuesta para evaluación.
    """
    params = {}
    # Para seguridad, no añadimos nada extra aquí; 'url' ya viene con query completa
    # Medición de alta resolución
    t_send = time.perf_counter_ns()
    try:
        with requests.get(url, stream=True, timeout=timeout) as r:
            status = r.status_code
            first_ns = None
            total_bytes = 0
            chunks = []
            for chunk in r.iter_content(chunk_size=8192):
                if not chunk:
                    continue
                if first_ns is None:
                    first_ns = time.perf_counter_ns()
                chunks.append(chunk)
                total_bytes += len(chunk)
            last_ns = time.perf_counter_ns()
            if first_ns is None:
                # No hubo cuerpo; consideramos primer byte al cerrar headers
                first_ns = last_ns
            ttfb_ms = (first_ns - t_send) / 1e6
            ttlb_ms = (last_ns - t_send) / 1e6
            
            # Reconstruir contenido completo
            content = b"".join(chunks).decode("utf-8", errors="replace")
            
            return {
                "status": status,
                "ttfb_ms": ttfb_ms,
                "ttlb_ms": ttlb_ms,
                "bytes": total_bytes,
                "timeout": 0,
                "content": content
            }
    except requests.exceptions.Timeout:
        return {"status": None, "ttfb_ms": None, "ttlb_ms": None, "bytes": 0, "timeout": 1, "content": ""}
    except requests.exceptions.ConnectionError:
        return {"status": None, "ttfb_ms": None, "ttlb_ms": None, "bytes": 0, "timeout": 1, "content": ""}
    except Exception as e:
        # Considerar como timeout/error
        return {"status": None, "ttfb_ms": None, "ttlb_ms": None, "bytes": 0, "timeout": 1, "content": ""}

def build_url(base_endpoint: str, query: str, model: str) -> str:
    # base_endpoint ya incluye /webapi/entry.cgi
    sep = "&" if "?" in base_endpoint else "?"
    return f"{base_endpoint}?{query}&model_name={model}"

# -----------------------
# Aggregation helpers
# -----------------------

def summarise_percentiles(df: pd.DataFrame) -> pd.DataFrame:
    """
    p50/p95/p99 for TTFB and TTLB + timeout rate + jitter per (model, pipeline, network)
    Jitter is calculated as:
    - Standard deviation (std)
    - Interquartile range (IQR = Q3 - Q1)
    """
    rows = []
    for (m, p, n), g in df.groupby(["model", "pipeline", "network"]):
        tot = len(g)
        tout = int(g["timeout"].sum())
        ok = g[g["timeout"] == 0]
        
        def pct(series, q):
            vals = series.dropna().values
            return np.percentile(vals, q) if len(vals) else np.nan
        
        def jitter_stats(series):
            """Calculate jitter metrics: std and IQR"""
            vals = series.dropna().values
            if len(vals) < 2:
                return np.nan, np.nan
            std = np.std(vals, ddof=1)  # sample std
            q1 = np.percentile(vals, 25)
            q3 = np.percentile(vals, 75)
            iqr = q3 - q1
            return std, iqr
        
        ttfb_jitter_std, ttfb_jitter_iqr = jitter_stats(ok["ttfb_ms"])
        ttlb_jitter_std, ttlb_jitter_iqr = jitter_stats(ok["ttlb_ms"])
        
        rows.append({
            "model": m,
            "pipeline": p,
            "network": n,
            "ttfb_p50_ms": pct(ok["ttfb_ms"], 50),
            "ttfb_p95_ms": pct(ok["ttfb_ms"], 95),
            "ttfb_p99_ms": pct(ok["ttfb_ms"], 99),
            "ttfb_jitter_std_ms": ttfb_jitter_std,
            "ttfb_jitter_iqr_ms": ttfb_jitter_iqr,
            "ttlb_p50_ms": pct(ok["ttlb_ms"], 50),
            "ttlb_p95_ms": pct(ok["ttlb_ms"], 95),
            "ttlb_p99_ms": pct(ok["ttlb_ms"], 99),
            "ttlb_jitter_std_ms": ttlb_jitter_std,
            "ttlb_jitter_iqr_ms": ttlb_jitter_iqr,
            "timeout_rate": tout / tot if tot else np.nan,
            "n": tot
        })
    return pd.DataFrame(rows)

def summarise_correctness(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula métricas de correctness promedio por (model, pipeline, network).
    Incluye: parse_rate, accuracy, avg_similarity, avg_key_match
    """
    rows = []
    for (m, p, n), g in df.groupby(["model", "pipeline", "network"]):
        # Filtrar solo respuestas exitosas (no timeout)
        ok = g[g["timeout"] == 0]
        
        if len(ok) == 0:
            rows.append({
                "model": m,
                "pipeline": p,
                "network": n,
                "parse_rate": 0.0,
                "accuracy": 0.0,
                "avg_similarity": 0.0,
                "avg_key_match": 0.0,
                "n_evaluated": 0
            })
            continue
        
        parse_rate = ok["ok_parse"].mean()
        accuracy = ok["class_ok"].mean()
        avg_similarity = ok["similarity"].mean()
        # key_match puede tener NaN, usar solo valores válidos
        key_match_vals = ok["key_match"].dropna()
        avg_key_match = key_match_vals.mean() if len(key_match_vals) > 0 else 0.0
        
        rows.append({
            "model": m,
            "pipeline": p,
            "network": n,
            "parse_rate": parse_rate,
            "accuracy": accuracy,
            "avg_similarity": avg_similarity,
            "avg_key_match": avg_key_match,
            "n_evaluated": len(ok)
        })
    
    return pd.DataFrame(rows)

# -----------------------
# Main experiment
# -----------------------

def main():
    parser = argparse.ArgumentParser(description="E1 Latency experiment: API-only vs System; LAN vs WAN")
    parser.add_argument("--runs-per-request", type=int, default=5, help="Repetitions per (request, model, pipeline, network)")
    parser.add_argument("--warmup", type=int, default=2, help="Warm-up requests per condition (not recorded)")
    parser.add_argument("--wan-delay-ms", type=int, default=80, help="WAN simulated delay (ms)")
    parser.add_argument("--wan-loss-pct", type=float, default=0.5, help="WAN simulated loss (%)")
    parser.add_argument("--timeout", type=float, default=60.0, help="HTTP timeout (s)")
    parser.add_argument("--skip-wan", action="store_true", help="Skip WAN simulation")
    parser.add_argument("--gold-file", type=str, default=GOLD_FILE, help="Path to gold.json file")
    parser.add_argument("--skip-correctness", action="store_true", help="Skip correctness evaluation")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    
    # Cargar gold responses si está disponible
    gold_map = {}
    if not args.skip_correctness:
        gold_map = load_gold_responses(args.gold_file)
        if not gold_map:
            print("[WARN] No gold responses loaded. Correctness metrics will be unavailable.")

    # Esquema CSV (con métricas de correctness)
    fieldnames = [
        "ts_iso", "run_id", "model", "pipeline", "network", "request_slug",
        "url", "status", "ttfb_ms", "ttlb_ms", "bytes", "timeout",
        "ok_parse", "class_ok", "dist_raw", "similarity", "key_match"
    ]
    with open(RAW_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    conditions = []
    # Pipelines
    pipelines = [("api", API_ENDPOINT_API), ("system", API_ENDPOINT_SYS)]
    # Networks
    networks = [("lan", None)]
    if not args.skip_wan:
        networks.append(("wan", {"delay_ms": args.wan_delay_ms, "loss_pct": args.wan_loss_pct}))

    run_id = 0
    for model in MODELS:
        for pipeline_name, base_url in pipelines:
            for net_name, net_cfg in networks:

                print(f"\n[INFO] Model: {model}, Pipeline: {pipeline_name}, Network: {net_name}")

                # Aplicar perfil de red si WAN
                applied = False
                if net_name == "wan":
                    applied = tc_apply("wan", delay_ms=args.wan_delay_ms, loss_pct=args.wan_loss_pct)
                    if not applied:
                        print("[WARN] WAN profile not active; continuing effectively as LAN.")

                # Warm-up
                for _ in range(args.warmup):
                    q = REQUESTS[0]
                    url = build_url(base_url, q, model)
                    _ = measure_http_stream(url, timeout=args.timeout)

                # Runs
                for q in REQUESTS:
                    slug = "".join([c if c.isalnum() else "_" for c in q])[:80]
                    for r in range(args.runs_per_request):
                        run_id += 1
                        url = build_url(base_url, q, model)
                        res = measure_http_stream(url, timeout=args.timeout)
                        
                        # Evaluar correctness si hay gold y no hubo timeout
                        correctness_metrics = {
                            "ok_parse": None,
                            "class_ok": None,
                            "dist_raw": None,
                            "similarity": None,
                            "key_match": None
                        }
                        
                        if not args.skip_correctness and gold_map and res["timeout"] == 0 and slug in gold_map:
                            correctness_metrics = evaluate_response(res["content"], gold_map[slug])
                        
                        row = {
                            "ts_iso": datetime.utcnow().isoformat(),
                            "run_id": run_id,
                            "model": model,
                            "pipeline": pipeline_name,
                            "network": net_name if (net_name != "wan" or applied) else "lan",  # si falla tc, etiqueta como lan
                            "request_slug": slug,
                            "url": url,
                            "status": res["status"],
                            "ttfb_ms": res["ttfb_ms"],
                            "ttlb_ms": res["ttlb_ms"],
                            "bytes": res["bytes"],
                            "timeout": res["timeout"],
                            "ok_parse": correctness_metrics["ok_parse"],
                            "class_ok": correctness_metrics["class_ok"],
                            "dist_raw": correctness_metrics["dist_raw"],
                            "similarity": correctness_metrics["similarity"],
                            "key_match": correctness_metrics["key_match"]
                        }
                        with open(RAW_CSV, "a", newline="", encoding="utf-8") as f:
                            writer = csv.DictWriter(f, fieldnames=fieldnames)
                            writer.writerow(row)

                # Limpiar netem si se aplicó
                if net_name == "wan" and applied:
                    tc_clear()

    # -----------------------
    # Agregación de resultados
    # -----------------------
    df = pd.read_csv(RAW_CSV)
    
    # Tabla resumen con percentiles y jitter
    summary = summarise_percentiles(df)
    summary.to_csv(SUMMARY_CSV, index=False)
    
    # Tabla resumen de correctness (si se evaluó)
    if not args.skip_correctness and gold_map:
        correctness_summary = summarise_correctness(df)
        correctness_summary.to_csv(CORRECTNESS_CSV, index=False)
        print("\n✅ Done.")
        print(f"• Raw CSV: {RAW_CSV}")
        print(f"• Latency summary (with jitter): {SUMMARY_CSV}")
        print(f"• Correctness summary: {CORRECTNESS_CSV}")
    else:
        print("\n✅ Done.")
        print(f"• Raw CSV: {RAW_CSV}")
        print(f"• Summary (with jitter): {SUMMARY_CSV}")
    
    print(f"\n📊 Note: Graphical representations will be generated in post-processing.")
    
    # Mostrar preview de métricas de correctness si están disponibles
    if not args.skip_correctness and gold_map and os.path.exists(CORRECTNESS_CSV):
        print("\n=== Correctness Metrics Preview ===")
        corr_df = pd.read_csv(CORRECTNESS_CSV)
        print(corr_df.round(3).to_string(index=False))

if __name__ == "__main__":
    main()

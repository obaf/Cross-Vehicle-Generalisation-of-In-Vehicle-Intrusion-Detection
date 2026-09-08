"""
run_benchmark.py -- the cross-vehicle generalisation benchmark (experiment1-by-fable).

    python src/run_benchmark.py --cache data_cache --work <local dir> --out results

Phase 1 featurises every vehicle domain once into a local feature store
(one .npy per array, memory-mapped by the workers). Phase 2 runs every
(protocol, fold, held-out vehicle, feature space) job in a process pool; each
job writes its own rows CSV and score archive, so an interrupted run resumes.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_THREADS = 4


def _init(threads):
    global _THREADS
    _THREADS = threads
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS"):
        os.environ[v] = str(threads)
    import warnings
    warnings.filterwarnings("ignore")


def _worker(spec, work, out, include_nn, include_ens, cap_train, fast=False):
    import bench_lib as BL
    domains = BL.load_domains(work)
    tag = f"{spec['protocol']}_f{spec['fold']}_{spec['target']}_{spec['fs']}"
    logf = open(os.path.join(out, "logs", tag + ".log"), "a")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n"); logf.flush()

    t0 = time.time()
    BL.run_job(spec, domains, _THREADS,
               os.path.join(out, "rows", tag + ".csv"),
               os.path.join(work, "scores", tag + ".npz"),
               include_nn=include_nn, include_ens=include_ens, cap_train=cap_train,
               fast=fast, log=log)
    log(f"[{tag}] done in {time.time()-t0:.0f}s")
    logf.close()
    return tag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data_cache")
    ap.add_argument("--work", required=True, help="local directory for features and scores")
    ap.add_argument("--out", default="results")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--cap-train", type=int, default=150_000)
    ap.add_argument("--vehicles", nargs="*", default=None)
    ap.add_argument("--feature-sets", nargs="+", default=["FS-A", "FS-B"])
    ap.add_argument("--protocols", nargs="+", default=["cross_domain", "in_domain"])
    ap.add_argument("--no-nn", action="store_true")
    ap.add_argument("--no-ens", action="store_true")
    ap.add_argument("--featurise-only", action="store_true")
    ap.add_argument("--fast", action="store_true", help="fast model subset (smoke test)")
    args = ap.parse_args()

    import bench_lib as BL
    os.makedirs(os.path.join(args.out, "rows"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "logs"), exist_ok=True)
    os.makedirs(os.path.join(args.work, "scores"), exist_ok=True)

    t_all = time.time()
    BL.featurise_all(args.cache, args.work, args.vehicles, workers=args.workers)
    if args.featurise_only:
        return
    domains = sorted(os.listdir(os.path.join(args.work, "features")))
    if args.vehicles:
        domains = [d for d in domains if d in args.vehicles]

    specs = []
    for target in domains:
        for fs in args.feature_sets:
            if "cross_domain" in args.protocols:
                specs.append(dict(protocol="cross_domain", fold=0, target=target, fs=fs))
            if "in_domain" in args.protocols:
                for fold in (0, 1):
                    specs.append(dict(protocol="in_domain", fold=fold, target=target, fs=fs))
    todo = [s for s in specs if not os.path.exists(os.path.join(
        args.out, "rows", f"{s['protocol']}_f{s['fold']}_{s['target']}_{s['fs']}.csv"))]
    print(f"{len(specs)} jobs, {len(todo)} to run, {args.workers} workers x {args.threads} threads",
          flush=True)

    from concurrent.futures import ProcessPoolExecutor, as_completed
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init,
                             initargs=(args.threads,)) as ex:
        futs = [ex.submit(_worker, s, args.work, args.out, not args.no_nn, not args.no_ens,
                          args.cap_train, args.fast) for s in todo]
        for f in as_completed(futs):
            try:
                print(f"== finished {f.result()}  ({time.time()-t_all:.0f}s elapsed)", flush=True)
            except Exception as e:      # noqa: BLE001
                print(f"== JOB FAILED: {type(e).__name__}: {e}", flush=True)

    import pandas as pd
    rows = [pd.read_csv(f) for f in sorted(glob.glob(os.path.join(args.out, "rows", "*.csv")))]
    df = pd.concat(rows, ignore_index=True)
    df.to_csv(os.path.join(args.out, "results_raw.csv"), index=False)
    print(f"\n{len(df)} result rows -> {args.out}/results_raw.csv  ({time.time()-t_all:.0f}s)",
          flush=True)


if __name__ == "__main__":
    main()

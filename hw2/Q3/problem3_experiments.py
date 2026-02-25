import argparse
import csv
import json
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np


TRAIN_N = 100
TEST_N = 1000
D = 100


@dataclass
class DataSplit:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    w_true: np.ndarray


def make_data(rng: np.random.Generator) -> DataSplit:
    X_train = rng.normal(0.0, 1.0, size=(TRAIN_N, D))
    w_true = rng.normal(0.0, 1.0, size=(D, 1))
    y_train = X_train @ w_true + rng.normal(0.0, 0.5, size=(TRAIN_N, 1))

    X_test = rng.normal(0.0, 1.0, size=(TEST_N, D))
    y_test = X_test @ w_true + rng.normal(0.0, 0.5, size=(TEST_N, 1))
    return DataSplit(X_train, y_train, X_test, y_test, w_true)


def normalized_error(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    resid = X @ w - y
    return float(np.linalg.norm(resid, 2) / np.linalg.norm(y, 2))


def ridge_solution(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    xtx = X.T @ X
    reg = lam * np.eye(X.shape[1])
    return np.linalg.solve(xtx + reg, X.T @ y)


def sgd_squared_loss(
    X: np.ndarray,
    y: np.ndarray,
    step_size: float,
    iters: int,
    rng: np.random.Generator,
    w0: np.ndarray | None = None,
    trace_every: int = 0,
    X_test: np.ndarray | None = None,
    y_test: np.ndarray | None = None,
    w_true: np.ndarray | None = None,
):
    n, d = X.shape
    w = np.zeros((d, 1)) if w0 is None else w0.copy()

    trace = {
        "steps": [],
        "train_err": [],
        "test_err": [],
        "norm": [],
        "true_train_err": None,
    }
    if w_true is not None:
        trace["true_train_err"] = normalized_error(X, y, w_true)

    for t in range(1, iters + 1):
        i = int(rng.integers(0, n))
        xi = X[i : i + 1, :]  # (1,d)
        yi = y[i : i + 1, :]  # (1,1)
        pred = xi @ w
        grad = 2.0 * (pred - yi) * xi.T
        w -= step_size * grad

        if trace_every and (t % trace_every == 0 or t == 1 or t == iters):
            trace["steps"].append(t)
            trace["train_err"].append(normalized_error(X, y, w))
            trace["norm"].append(float(np.linalg.norm(w, 2)))
            if X_test is not None and y_test is not None:
                trace["test_err"].append(normalized_error(X_test, y_test, w))

    return w, trace


def write_csv(path: str, header: list[str], rows: list[list[object]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def run_all(args: argparse.Namespace) -> dict:
    os.makedirs(args.outdir, exist_ok=True)

    lams = [0.0005, 0.005, 0.05, 0.5, 5, 50, 500]
    step_sizes_33 = [0.00005, 0.0005, 0.005]
    step_sizes_34 = [0.00005, 0.005]
    radii_35 = [0, 0.1, 0.5, 1, 10, 20, 30]

    # 3.1
    tr_err_31 = []
    te_err_31 = []
    for k in range(args.trials):
        rng = np.random.default_rng(args.seed + 1000 + k)
        ds = make_data(rng)
        w = np.linalg.solve(ds.X_train, ds.y_train)
        tr_err_31.append(normalized_error(ds.X_train, ds.y_train, w))
        te_err_31.append(normalized_error(ds.X_test, ds.y_test, w))

    mean_31 = {
        "train": float(np.mean(tr_err_31)),
        "test": float(np.mean(te_err_31)),
    }

    # 3.2
    ridge_stats = {lam: {"train": [], "test": []} for lam in lams}
    for k in range(args.trials):
        rng = np.random.default_rng(args.seed + 2000 + k)
        ds = make_data(rng)
        for lam in lams:
            w = ridge_solution(ds.X_train, ds.y_train, lam)
            ridge_stats[lam]["train"].append(normalized_error(ds.X_train, ds.y_train, w))
            ridge_stats[lam]["test"].append(normalized_error(ds.X_test, ds.y_test, w))

    rows_32 = []
    ridge_mean_train = []
    ridge_mean_test = []
    for lam in lams:
        mtr = float(np.mean(ridge_stats[lam]["train"]))
        mte = float(np.mean(ridge_stats[lam]["test"]))
        ridge_mean_train.append(mtr)
        ridge_mean_test.append(mte)
        rows_32.append([lam, mtr, mte])
    write_csv(os.path.join(args.outdir, "table_3_2_ridge.csv"), ["lambda", "train_err", "test_err"], rows_32)

    plt.figure(figsize=(7, 4.5))
    plt.semilogx(lams, ridge_mean_train, marker="o", label="train")
    plt.semilogx(lams, ridge_mean_test, marker="o", label="test")
    plt.xlabel("lambda")
    plt.ylabel("normalized error")
    plt.title("Problem 3.2: Ridge train/test error vs lambda")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "fig_3_2_ridge_curve.png"), dpi=160)
    plt.close()

    # 3.3
    rows_33 = []
    sgd33 = {}
    for eta in step_sizes_33:
        tr_list = []
        te_list = []
        for k in range(args.trials):
            rng = np.random.default_rng(args.seed + 3000 + k)
            ds = make_data(rng)
            w, _ = sgd_squared_loss(ds.X_train, ds.y_train, eta, args.iters_33, rng)
            tr_list.append(normalized_error(ds.X_train, ds.y_train, w))
            te_list.append(normalized_error(ds.X_test, ds.y_test, w))
        mtr = float(np.mean(tr_list))
        mte = float(np.mean(te_list))
        rows_33.append([eta, mtr, mte])
        sgd33[str(eta)] = {"train": mtr, "test": mte}
    write_csv(os.path.join(args.outdir, "table_3_3_sgd.csv"), ["step_size", "train_err", "test_err"], rows_33)

    # 3.4 (single trial traces)
    rng = np.random.default_rng(args.seed + 4000)
    ds = make_data(rng)

    traces_34 = {}
    for eta in step_sizes_34:
        _, trace = sgd_squared_loss(
            ds.X_train,
            ds.y_train,
            eta,
            args.iters_34,
            rng,
            trace_every=args.trace_every,
            X_test=ds.X_test,
            y_test=ds.y_test,
            w_true=ds.w_true,
        )
        traces_34[str(eta)] = trace

    plt.figure(figsize=(7, 4.5))
    for eta in step_sizes_34:
        tr = traces_34[str(eta)]
        plt.plot(tr["steps"], tr["train_err"], label=f"eta={eta}")
    plt.axhline(traces_34[str(step_sizes_34[0])]["true_train_err"], linestyle="--", label="true model train err")
    plt.xlabel("iteration")
    plt.ylabel("normalized train error")
    plt.title("Problem 3.4(i): Train error trajectory")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "fig_3_4_train_curve.png"), dpi=160)
    plt.close()

    plt.figure(figsize=(7, 4.5))
    for eta in step_sizes_34:
        tr = traces_34[str(eta)]
        plt.plot(tr["steps"], tr["test_err"], label=f"eta={eta}")
    plt.xlabel("iteration")
    plt.ylabel("normalized test error")
    plt.title("Problem 3.4(ii): Test error trajectory")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "fig_3_4_test_curve.png"), dpi=160)
    plt.close()

    plt.figure(figsize=(7, 4.5))
    for eta in step_sizes_34:
        tr = traces_34[str(eta)]
        plt.plot(tr["steps"], tr["norm"], label=f"eta={eta}")
    plt.xlabel("iteration")
    plt.ylabel("||w_t||_2")
    plt.title("Problem 3.4(iii): Parameter norm trajectory")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "fig_3_4_norm_curve.png"), dpi=160)
    plt.close()

    # 3.5
    rows_35 = []
    for r in radii_35:
        tr_list = []
        te_list = []
        for k in range(args.trials):
            rng = np.random.default_rng(args.seed + 5000 + k)
            ds = make_data(rng)

            if r == 0:
                w0 = np.zeros((D, 1))
            else:
                g = rng.normal(0.0, 1.0, size=(D, 1))
                g = g / np.linalg.norm(g, 2)
                w0 = r * g

            w, _ = sgd_squared_loss(ds.X_train, ds.y_train, 0.00005, args.iters_35, rng, w0=w0)
            tr_list.append(normalized_error(ds.X_train, ds.y_train, w))
            te_list.append(normalized_error(ds.X_test, ds.y_test, w))

        mtr = float(np.mean(tr_list))
        mte = float(np.mean(te_list))
        rows_35.append([r, mtr, mte])

    write_csv(os.path.join(args.outdir, "table_3_5_init_radius.csv"), ["radius", "train_err", "test_err"], rows_35)

    rr = [row[0] for row in rows_35]
    trr = [row[1] for row in rows_35]
    ter = [row[2] for row in rows_35]
    plt.figure(figsize=(7, 4.5))
    plt.plot(rr, trr, marker="o", label="train")
    plt.plot(rr, ter, marker="o", label="test")
    plt.xlabel("init radius r")
    plt.ylabel("normalized error")
    plt.title("Problem 3.5: Error vs initialization radius")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "fig_3_5_init_radius_curve.png"), dpi=160)
    plt.close()

    summary = {
        "3.1": mean_31,
        "3.2": rows_32,
        "3.3": sgd33,
        "3.5": rows_35,
        "config": {
            "trials": args.trials,
            "iters_33": args.iters_33,
            "iters_34": args.iters_34,
            "iters_35": args.iters_35,
            "trace_every": args.trace_every,
            "seed": args.seed,
        },
    }

    with open(os.path.join(args.outdir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Problem 3 experiments (HW2 CSCI567)")
    parser.add_argument("--outdir", type=str, default="results", help="output directory")
    parser.add_argument("--trials", type=int, default=10, help="number of trials for averaged parts")
    parser.add_argument("--iters-33", type=int, default=1_000_000, dest="iters_33")
    parser.add_argument("--iters-34", type=int, default=1_000_000, dest="iters_34")
    parser.add_argument("--iters-35", type=int, default=1_000_000, dest="iters_35")
    parser.add_argument("--trace-every", type=int, default=100, dest="trace_every")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_all(args)
    print("Done. Summary:")
    print(json.dumps(summary["3.1"], indent=2))


if __name__ == "__main__":
    main()

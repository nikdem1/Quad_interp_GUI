import ast
import math
import threading
import time
import csv
from dataclasses import dataclass
from typing import Callable, List, Tuple, Optional

import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


# ------------------------- safe expression evaluator -------------------------

_ALLOWED_NAMES = {
    # constants
    "pi": math.pi,
    "e": math.e,

    # math funcs
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "sinh": math.sinh,
    "cosh": math.cosh,
    "tanh": math.tanh,
    "exp": math.exp,
    "log": math.log,
    "log10": math.log10,
    "sqrt": math.sqrt,
    "fabs": math.fabs,
    "floor": math.floor,
    "ceil": math.ceil,
    "pow": pow,
}

_ALLOWED_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Num,
    ast.Constant,
    ast.Name,
    ast.Load,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod,
    ast.USub, ast.UAdd,
    ast.Call,
)


def _compile_expr(expr: str, var_names: List[str]) -> Callable[[np.ndarray], float]:
    tree = ast.parse(expr, mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ValueError(f"Недопустимое выражение: {type(node).__name__}")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_NAMES:
                raise ValueError("Разрешены только функции из math: sin, cos, exp, log, sqrt, ...")
        if isinstance(node, ast.Name):
            if node.id not in _ALLOWED_NAMES and node.id not in var_names:
                raise ValueError(f"Неизвестное имя: {node.id}")

    code = compile(tree, "<expr>", "eval")

    def f(x: np.ndarray) -> float:
        env = dict(_ALLOWED_NAMES)
        for i, vn in enumerate(var_names):
            env[vn] = float(x[i])
        return float(eval(code, {"__builtins__": {}}, env))

    return f


def infer_vars(expr: str) -> List[str]:
    """Try infer vars among x, y, z, w, x1..x10. Prefer x,y,..."""
    candidates = ["x", "y", "z", "w"] + [f"x{i}" for i in range(1, 11)]
    found = set()
    tree = ast.parse(expr, mode="eval")
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in candidates:
            found.add(node.id)
    # Order: x,y,z,w then x1..x10
    ordered = [c for c in candidates if c in found]
    if not ordered:
        ordered = ["x"]  # default 1D
    return ordered


# ------------------------- quadratic interpolation (1D line search) -------------------------

@dataclass
class StepRecord:
    k: int
    x: np.ndarray
    fx: float
    info: str


def quad_interp_1d(phi: Callable[[float], float],
                   t0: float = 0.0,
                   h: float = 0.2,
                   tol: float = 1e-6,
                   max_iter: int = 50) -> Tuple[float, float, List[Tuple[float, float]]]:
    """
    1D parabolic (quadratic) interpolation using 3 points.
    Returns best_t, best_phi, and trace list of (t, phi(t)) for this line-search.
    """
    trace = []

    t1 = t0
    f1 = phi(t1); trace.append((t1, f1))

    t2 = t1 + h
    f2 = phi(t2); trace.append((t2, f2))

    # If uphill, reverse direction
    if f2 > f1:
        h = -h
        t2 = t1 + h
        f2 = phi(t2); trace.append((t2, f2))

    t3 = t1 + 2*h
    f3 = phi(t3); trace.append((t3, f3))

    best_t, best_f = t1, f1
    for t, ft in [(t1, f1), (t2, f2), (t3, f3)]:
        if ft < best_f:
            best_t, best_f = t, ft

    for _ in range(max_iter):
        # Parabola vertex formula
        # Use t2 as middle point (not required, but stable when points are spaced)
        denom = (t2 - t1) * (f2 - f3) - (t2 - t3) * (f2 - f1)
        if abs(denom) < 1e-14:
            break

        numer = ((t2 - t1) ** 2) * (f2 - f3) - ((t2 - t3) ** 2) * (f2 - f1)
        t_star = t2 - 0.5 * numer / denom

        f_star = phi(t_star)
        trace.append((t_star, f_star))

        # Update best
        if f_star < best_f:
            best_t, best_f = t_star, f_star

        # Rebuild triple around best: pick 3 closest points by t distance from best
        pts = sorted([(t1, f1), (t2, f2), (t3, f3), (t_star, f_star)], key=lambda p: abs(p[0] - best_t))
        # Take best itself + two nearest distinct t
        new_pts = []
        seen = set()
        for t, ft in pts:
            key = round(t, 14)
            if key in seen:
                continue
            seen.add(key)
            new_pts.append((t, ft))
            if len(new_pts) == 3:
                break

        if len(new_pts) < 3:
            break

        # Sort by t for next iteration
        new_pts.sort(key=lambda p: p[0])
        (t1, f1), (t2, f2), (t3, f3) = new_pts

        # Stop check: tight bracket & improvement
        if max(abs(t2 - t1), abs(t3 - t2)) < tol:
            break

    return best_t, best_f, trace


# ------------------------- nD minimization via coordinate descent + quad interp -------------------------

def coord_descent_quad(f: Callable[[np.ndarray], float],
                       x0: np.ndarray,
                       h0: float,
                       tol: float,
                       max_iter: int,
                       stop_flag: threading.Event,
                       per_line_max_iter: int = 30) -> List[StepRecord]:
    """
    Coordinate descent (Gauss-Seidel style): for each coordinate i, minimize along that axis
    using 1D quadratic interpolation for step t, then update x_i.
    """
    x = x0.astype(float).copy()
    n = len(x)
    records: List[StepRecord] = []
    fx = f(x)
    records.append(StepRecord(0, x.copy(), fx, "start"))

    k = 0
    for it in range(1, max_iter + 1):
        if stop_flag.is_set():
            records.append(StepRecord(k, x.copy(), fx, "stopped"))
            break

        x_prev = x.copy()
        fx_prev = fx

        for i in range(n):
            if stop_flag.is_set():
                records.append(StepRecord(k, x.copy(), fx, "stopped"))
                return records

            # line function along coordinate i: phi(t) = f(x + t*e_i)
            def phi(t: float, i=i, xbase=x.copy()):
                xt = xbase.copy()
                xt[i] = xbase[i] + t
                return f(xt)

            best_t, best_phi, _ = quad_interp_1d(phi, t0=0.0, h=h0, tol=tol, max_iter=per_line_max_iter)
            x[i] = x[i] + best_t
            fx = f(x)
            k += 1
            records.append(StepRecord(k, x.copy(), fx, f"coord {i+1}, t*={best_t:.6g}"))

        # stopping by point change + f change
        if np.linalg.norm(x - x_prev) < tol and abs(fx - fx_prev) < tol:
            records.append(StepRecord(k, x.copy(), fx, "converged"))
            break

    return records


# ------------------------- GUI -------------------------

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Quadratic Interpolation — минимизация (GUI)")
        self.geometry("1180x720")

        self.stop_flag = threading.Event()
        self.worker: Optional[threading.Thread] = None
        self.records: List[StepRecord] = []

        self._build_ui()
        self._build_plot()

    def _build_ui(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        top = ttk.Frame(self, padding=10)
        top.grid(row=0, column=0, sticky="ew")
        top.columnconfigure(1, weight=1)

        ttk.Label(top, text="Функция (переменные: x, y, ...):").grid(row=0, column=0, sticky="w")
        self.func_var = tk.StringVar(value="x**2 + y**2")
        ttk.Entry(top, textvariable=self.func_var).grid(row=0, column=1, sticky="ew", padx=8)

        ttk.Label(top, text="Начальная точка (comma-separated):").grid(row=1, column=0, sticky="w")
        self.x0_var = tk.StringVar(value="1, 1")
        ttk.Entry(top, textvariable=self.x0_var).grid(row=1, column=1, sticky="ew", padx=8)

        ttk.Label(top, text="Начальный шаг (h0):").grid(row=2, column=0, sticky="w")
        self.h0_var = tk.StringVar(value="0.5")
        ttk.Entry(top, textvariable=self.h0_var, width=12).grid(row=2, column=1, sticky="w", padx=8)

        ttk.Label(top, text="Макс. итераций (циклы):").grid(row=3, column=0, sticky="w")
        self.max_iter_var = tk.StringVar(value="50")
        ttk.Entry(top, textvariable=self.max_iter_var, width=12).grid(row=3, column=1, sticky="w", padx=8)

        ttk.Label(top, text="Точность (tol):").grid(row=4, column=0, sticky="w")
        self.tol_var = tk.StringVar(value="1e-6")
        ttk.Entry(top, textvariable=self.tol_var, width=12).grid(row=4, column=1, sticky="w", padx=8)

        btns = ttk.Frame(top)
        btns.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        btns.columnconfigure(0, weight=1)
        btns.columnconfigure(1, weight=1)
        btns.columnconfigure(2, weight=1)

        self.start_btn = ttk.Button(btns, text="Start", command=self.on_start)
        self.start_btn.grid(row=0, column=0, sticky="ew", padx=5)

        self.stop_btn = ttk.Button(btns, text="Stop", command=self.on_stop, state="disabled")
        self.stop_btn.grid(row=0, column=1, sticky="ew", padx=5)

        self.export_btn = ttk.Button(btns, text="Export CSV", command=self.on_export, state="disabled")
        self.export_btn.grid(row=0, column=2, sticky="ew", padx=5)

        mid = ttk.Frame(self, padding=(10, 0, 10, 10))
        mid.grid(row=1, column=0, sticky="nsew")
        mid.columnconfigure(0, weight=2)
        mid.columnconfigure(1, weight=1)
        mid.rowconfigure(0, weight=1)

        self.plot_frame = ttk.Frame(mid)
        self.plot_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        right = ttk.Frame(mid)
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(1, weight=1)

        ttk.Label(right, text="Лог итераций:").grid(row=0, column=0, sticky="w")
        self.log = tk.Text(right, height=10, wrap="none")
        self.log.grid(row=1, column=0, sticky="nsew")

        self.status_var = tk.StringVar(value="Готово")
        ttk.Label(self, textvariable=self.status_var, anchor="w", padding=6).grid(row=2, column=0, sticky="ew")

    def _build_plot(self):
        self.fig = Figure(figsize=(6.5, 4.8), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Итерационный процесс")
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y / f(x)")
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvas.draw()

    def log_line(self, s: str):
        self.log.insert("end", s + "\n")
        self.log.see("end")

    def on_start(self):
        if self.worker and self.worker.is_alive():
            return

        try:
            expr = self.func_var.get().strip()
            var_names = infer_vars(expr)

            x0 = np.array([float(v.strip()) for v in self.x0_var.get().split(",") if v.strip() != ""], dtype=float)
            if len(x0) == 0:
                raise ValueError("Пустая начальная точка.")
            if len(var_names) != len(x0):
                # allow 1D if only x specified but user gave 2 coords etc.
                # In that case, set var_names = x,y,z,w... length = len(x0)
                base = ["x", "y", "z", "w"] + [f"x{i}" for i in range(1, 11)]
                var_names = base[:len(x0)]

            h0 = float(self.h0_var.get())
            tol = float(self.tol_var.get())
            max_iter = int(float(self.max_iter_var.get()))
            if max_iter <= 0:
                raise ValueError("max_iter должен быть > 0")
            if tol <= 0:
                raise ValueError("tol должен быть > 0")
            if h0 == 0:
                raise ValueError("h0 не должен быть 0")

            f = _compile_expr(expr, var_names)

        except Exception as e:
            messagebox.showerror("Ошибка ввода", str(e))
            return

        # clear
        self.stop_flag.clear()
        self.records = []
        self.log.delete("1.0", "end")
        self.ax.clear()
        self.canvas.draw()

        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.export_btn.config(state="disabled")
        self.status_var.set("Выполняется...")

        def worker():
            try:
                recs = coord_descent_quad(
                    f=f,
                    x0=x0,
                    h0=h0,
                    tol=tol,
                    max_iter=max_iter,
                    stop_flag=self.stop_flag,
                    per_line_max_iter=40
                )
                self.records = recs
            except Exception as e:
                self.records = []
                self.after(0, lambda: messagebox.showerror("Ошибка выполнения", str(e)))
            finally:
                self.after(0, self.on_finish)

        self.worker = threading.Thread(target=worker, daemon=True)
        self.worker.start()

        # periodic UI updates while running
        self.after(200, self._poll_progress)

    def _poll_progress(self):
        if self.worker and self.worker.is_alive():
            # show last record if exists
            if self.records:
                r = self.records[-1]
                self.status_var.set(f"Итерация: {r.k}, x={np.array2string(r.x, precision=4)}, f={r.fx:.6g}")
            self.after(200, self._poll_progress)

    def on_stop(self):
        self.stop_flag.set()
        self.status_var.set("Останавливаем...")

    def on_finish(self):
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.export_btn.config(state="normal" if self.records else "disabled")

        if not self.records:
            self.status_var.set("Ошибка/нет данных")
            return

        # log output
        self.log_line("k | x | f(x) | info")
        for r in self.records[:1]:
            self.log_line(f"{r.k:>3d} | {np.array2string(r.x, precision=6)} | {r.fx:.6g} | {r.info}")
        if len(self.records) > 1:
            for r in self.records[1:]:
                self.log_line(f"{r.k:>3d} | {np.array2string(r.x, precision=6)} | {r.fx:.6g} | {r.info}")

        # plot
        self._plot_records()

        last = self.records[-1]
        self.status_var.set(f"Готово. Последняя точка: x={np.array2string(last.x, precision=6)}, f={last.fx:.6g}")

    def _plot_records(self):
        self.ax.clear()

        xs = np.array([r.x for r in self.records])
        fs = np.array([r.fx for r in self.records])
        dim = xs.shape[1]

        if dim == 1:
            # 1D plot: f(x) curve and visited points
            x_min = xs[:, 0].min()
            x_max = xs[:, 0].max()
            span = (x_max - x_min) if x_max > x_min else 1.0
            grid = np.linspace(x_min - 0.5*span, x_max + 0.5*span, 400)
            # try evaluate using stored expression by rebuilding? easiest: approximate using points line:
            # We'll just plot visited f values vs iteration and points on x-axis.
            self.ax.plot(range(len(fs)), fs, marker="o", linewidth=1.5)
            self.ax.set_title("f на итерациях (1D)")
            self.ax.set_xlabel("итерация")
            self.ax.set_ylabel("f(x)")
        else:
            # 2D contour + trajectory
            x1 = xs[:, 0]
            x2 = xs[:, 1]

            x1_min, x1_max = x1.min(), x1.max()
            x2_min, x2_max = x2.min(), x2.max()
            pad1 = max(1.0, 0.3 * (x1_max - x1_min + 1e-9))
            pad2 = max(1.0, 0.3 * (x2_max - x2_min + 1e-9))

            gx = np.linspace(x1_min - pad1, x1_max + pad1, 220)
            gy = np.linspace(x2_min - pad2, x2_max + pad2, 220)
            X, Y = np.meshgrid(gx, gy)

            # rebuild function from current input (safe)
            expr = self.func_var.get().strip()
            var_names = infer_vars(expr)
            if len(var_names) != 2:
                var_names = ["x", "y"]
            f = _compile_expr(expr, var_names)

            Z = np.zeros_like(X)
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    Z[i, j] = f(np.array([X[i, j], Y[i, j]]))

            self.ax.contour(X, Y, Z, levels=30)
            self.ax.plot(x1, x2, marker="o", linewidth=1.5, label="траектория")
            self.ax.scatter([x1[-1]], [x2[-1]], marker="x", s=80, label="последняя")
            self.ax.set_title("Контуры + траектория (2D)")
            self.ax.set_xlabel("x")
            self.ax.set_ylabel("y")
            self.ax.legend()

        self.ax.grid(True)
        self.canvas.draw()

    def on_export(self):
        if not self.records:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            title="Сохранить CSV"
        )
        if not path:
            return
        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f, delimiter=";")
                # header
                dim = len(self.records[0].x)
                header = ["k"] + [f"x{i+1}" for i in range(dim)] + ["f", "info"]
                w.writerow(header)
                for r in self.records:
                    w.writerow([r.k] + [f"{v:.12g}" for v in r.x.tolist()] + [f"{r.fx:.12g}", r.info])
            messagebox.showinfo("Готово", f"CSV сохранён:\n{path}")
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))


if __name__ == "__main__":
    app = App()
    app.mainloop()
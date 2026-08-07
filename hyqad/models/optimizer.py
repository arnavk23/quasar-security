import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize, differential_evolution, basinhopping


class JSOOptimizer:
    """
    Optimization routines inspired by JuliaSmoothOptimizers (JSO)
    internship work. Provides structured optimization for model
    hyperparameter tuning in the QKD anomaly detection pipeline.

    Supports L-BFGS-B, Nelder-Mead, differential evolution, and
    basin-hopping strategies with proper parameter bounds handling.
    """

    def __init__(
        self,
        method: str = "L-BFGS-B",
        max_iter: int = 500,
        n_restarts: int = 5,
        tol: float = 1e-6,
        random_state: int = 42,
    ):
        self.method = method
        self.max_iter = max_iter
        self.n_restarts = n_restarts
        self.tol = tol
        self.rng = np.random.default_rng(random_state)

    def _wrap_objective(self, objective_fn, bounds):
        def wrapped(params):
            clipped = np.clip(params, [b[0] for b in bounds], [b[1] for b in bounds])
            return objective_fn(clipped)
        return wrapped

    def optimize(self, objective_fn, bounds: list[tuple[float, float]],
                 x0: NDArray | None = None) -> tuple[NDArray, float]:
        best_x, best_f = None, np.inf

        if self.method == "differential_evolution":
            result = differential_evolution(
                objective_fn, bounds, maxiter=self.max_iter,
                tol=self.tol, seed=self.rng.integers(2**31),
                polish=True,
            )
            return result.x, result.fun

        if self.method == "basinhopping":
            if x0 is None:
                x0 = np.array([(b[0] + b[1]) / 2 for b in bounds])
            wrapped = self._wrap_objective(objective_fn, bounds)
            result = basinhopping(
                wrapped, x0, niter=self.max_iter,
                stepsize=0.1, T=1.0,
                minimizer_kwargs={"method": "L-BFGS-B", "bounds": bounds},
            )
            return result.x, result.fun

        for restart in range(self.n_restarts):
            if x0 is not None and restart == 0:
                start = np.asarray(x0, dtype=np.float64)
            else:
                start = np.array([
                    self.rng.uniform(b[0], b[1]) for b in bounds
                ])

            result = minimize(
                objective_fn, start, method=self.method,
                bounds=bounds,
                options={"maxiter": self.max_iter, "ftol": self.tol},
            )

            if result.fun < best_f:
                best_x, best_f = result.x.copy(), result.fun

        return best_x, best_f

    def line_search(self, objective_fn, x: NDArray, direction: NDArray,
                    bounds: list[tuple[float, float]],
                    max_steps: int = 20) -> float:
        step = 1.0
        for _ in range(max_steps):
            candidate = np.clip(x + step * direction,
                                [b[0] for b in bounds], [b[1] for b in bounds])
            if objective_fn(candidate) < objective_fn(x):
                return step
            step *= 0.5
        return 0.0

    def grid_search_cv(self, objective_fn, param_grid: dict[str, list],
                       fixed_params: dict | None = None) -> tuple[dict, float]:
        from itertools import product
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        best_params, best_score = None, np.inf

        for combo in product(*values):
            params = dict(zip(keys, combo))
            if fixed_params:
                params.update(fixed_params)
            score = objective_fn(params)
            if score < best_score:
                best_score = score
                best_params = params.copy()

        return best_params, best_score

    def tune_model_parameters(
        self,
        model,
        X_train: NDArray, y_train: NDArray,
        X_val: NDArray, y_val: NDArray,
        param_bounds: dict[str, tuple[float, float]],
        metric_fn=None,
    ) -> dict[str, float]:
        if metric_fn is None:
            def metric_fn(y_true, y_pred):
                from sklearn.metrics import accuracy_score
                return -accuracy_score(y_true, y_pred)

        param_names = list(param_bounds.keys())
        bounds = list(param_bounds.values())

        def objective(params_vec):
            for name, val in zip(param_names, params_vec):
                setattr(model, name, val)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            return metric_fn(y_val, y_pred)

        best_vec, _ = self.optimize(objective, bounds)
        return dict(zip(param_names, best_vec))

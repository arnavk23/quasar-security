import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC

try:
    import pennylane as qml
    HAS_PENNYLANE = True
except ImportError:
    HAS_PENNYLANE = False


def _pad_normalize(X: NDArray, dim: int) -> NDArray:
    """Zero-pad rows of X to ``dim`` and L2-normalize (the amplitude-encoding map)."""
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    if X.shape[1] > dim:
        raise ValueError(f"{X.shape[1]} features do not fit in a {dim}-amplitude register")
    padded = np.zeros((X.shape[0], dim))
    padded[:, :X.shape[1]] = X
    norms = np.linalg.norm(padded, axis=1, keepdims=True)
    zero = norms[:, 0] < 1e-12
    padded[zero, 0] = 1.0
    norms[zero] = 1.0
    return padded / norms


def amplitude_fidelity_kernel(X1: NDArray, X2: NDArray, n_qubits: int) -> NDArray:
    """Exact fidelity kernel |<psi(x1)|psi(x2)>|^2 for amplitude encoding.

    For amplitude-encoded states this is the squared cosine similarity of the
    zero-padded inputs, so it is classically computable in O(n1 * n2 * 2^q).
    It is identical (to float precision) to the PennyLane overlap circuit
    computed by ``QuantumKernelEstimator(backend="circuit")``.
    """
    dim = 2 ** n_qubits
    return (_pad_normalize(X1, dim) @ _pad_normalize(X2, dim).T) ** 2


class QuantumKernelEstimator(BaseEstimator, ClassifierMixin):
    """
    Fidelity-kernel SVM with quantum state encodings.

    kernel_type:
    - 'amplitude': amplitude encoding on ceil(log2 d) qubits (all d features
      are encoded; nothing is truncated).
    - 'angle': per-feature RX/RY angle encoding on d qubits.

    backend:
    - 'exact': closed-form classical evaluation (amplitude only).
    - 'circuit': PennyLane state-vector simulation of the overlap circuit.
      Circuit failures raise; there is no silent classical fallback.
    """

    def __init__(
        self,
        n_qubits: int | None = None,
        kernel_type: str = "amplitude",
        backend: str = "exact",
        C: float = 1.0,
        class_weight: str | None = "balanced",
        random_state: int = 42,
    ):
        self.n_qubits = n_qubits
        self.kernel_type = kernel_type
        self.backend = backend
        self.C = C
        self.class_weight = class_weight
        self.random_state = random_state

    def _resolve_qubits(self, d: int) -> int:
        if self.kernel_type == "angle":
            needed = d
        else:
            needed = max(1, int(np.ceil(np.log2(max(d, 2)))))
        if self.n_qubits is not None and self.n_qubits < needed:
            raise ValueError(f"{self.kernel_type} encoding of {d} features needs {needed} qubits, "
                             f"got n_qubits={self.n_qubits}")
        return self.n_qubits or needed

    def _build_circuit(self):
        if not HAS_PENNYLANE:
            raise ImportError("PennyLane is required for backend='circuit'.")
        q = self.n_qubits_
        dev = qml.device("default.qubit", wires=q)
        wires = range(q)

        if self.kernel_type == "amplitude":
            def embed(x):
                qml.StatePrep(x, wires=wires)
        elif self.kernel_type == "angle":
            def embed(x):
                for i in range(q):
                    qml.RX(x[i], wires=i)
                    qml.RY(x[i] ** 2, wires=i)
        else:
            raise ValueError(f"Unknown kernel_type: {self.kernel_type}")

        @qml.qnode(dev)
        def overlap(x1, x2):
            embed(x1)
            qml.adjoint(embed)(x2)
            return qml.probs(wires=wires)

        return overlap

    def _quantum_kernel(self, X1: NDArray, X2: NDArray) -> NDArray:
        if self.backend == "exact":
            if self.kernel_type != "amplitude":
                raise ValueError("backend='exact' is implemented for amplitude encoding only")
            return amplitude_fidelity_kernel(X1, X2, self.n_qubits_)
        if self.backend != "circuit":
            raise ValueError(f"Unknown backend: {self.backend}")

        if self.kernel_type == "amplitude":
            X1 = _pad_normalize(X1, 2 ** self.n_qubits_)
            X2 = _pad_normalize(X2, 2 ** self.n_qubits_)
        circuit = self._build_circuit()
        K = np.empty((len(X1), len(X2)))
        for i, x1 in enumerate(X1):
            for j, x2 in enumerate(X2):
                K[i, j] = circuit(x1, x2)[0]
        return K

    def fit(self, X: NDArray, y: NDArray):
        self.X_fit_ = np.asarray(X, dtype=np.float64)
        self.n_qubits_ = self._resolve_qubits(self.X_fit_.shape[1])
        K = self._quantum_kernel(self.X_fit_, self.X_fit_)
        self.svm_ = SVC(kernel="precomputed", C=self.C, probability=True,
                        class_weight=self.class_weight, random_state=self.random_state)
        self.svm_.fit(K, y)
        self.classes_ = self.svm_.classes_
        return self

    def decision_function(self, X: NDArray) -> NDArray:
        return self.svm_.decision_function(self._quantum_kernel(np.asarray(X, dtype=np.float64), self.X_fit_))

    def predict(self, X: NDArray) -> NDArray:
        K = self._quantum_kernel(np.asarray(X, dtype=np.float64), self.X_fit_)
        return self.svm_.predict(K)

    def predict_proba(self, X: NDArray) -> NDArray:
        K = self._quantum_kernel(np.asarray(X, dtype=np.float64), self.X_fit_)
        return self.svm_.predict_proba(K)

    def predict_with_uncertainty(self, X: NDArray) -> tuple[NDArray, NDArray, NDArray]:
        proba = self.predict_proba(X)
        anomaly_proba = proba[:, 1]
        predictions = (anomaly_proba >= 0.5).astype(int)
        uncertainty = np.sqrt(anomaly_proba * (1 - anomaly_proba))
        return predictions, anomaly_proba, uncertainty

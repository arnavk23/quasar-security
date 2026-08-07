import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC

try:
    import pennylane as qml
    HAS_PENNYLANE = True
except ImportError:
    HAS_PENNYLANE = False


class QuantumKernelEstimator(BaseEstimator, ClassifierMixin):
    """
    Quantum kernel method for anomaly detection using PennyLane.
    Embeds classical QKD features into quantum Hilbert space and
    computes quantum kernels for SVM classification.

    Supports three kernel types:
    - 'angle': Angle encoding with rotational gates
    - 'amplitude': Amplitude encoding
    - 'gaussian': Classical RBF fallback
    """

    def __init__(
        self,
        n_qubits: int = 3,
        kernel_type: str = "amplitude",
        gamma: float = 1.0,
        C: float = 1.0,
        n_restarts_optimizer: int = 5,
        random_state: int = 42,
    ):
        self.n_qubits = n_qubits
        self.kernel_type = kernel_type
        self.gamma = gamma
        self.C = C
        self.n_restarts_optimizer = n_restarts_optimizer
        self.random_state = random_state

    def _build_quantum_device(self):
        if not HAS_PENNYLANE:
            raise ImportError("PennyLane is required for quantum kernel methods.")
        return qml.device("default.qubit", wires=self.n_qubits)

    def _angle_embedding(self, x: NDArray, wires):
        for i in range(min(len(x), self.n_qubits)):
            qml.RX(x[i], wires=wires[i])
            qml.RY(x[i] ** 2, wires=wires[i])

    def _amplitude_embedding(self, x: NDArray, wires):
        n = min(2**self.n_qubits, len(x))
        padded = np.zeros(2**self.n_qubits)
        padded[:n] = x[:n]
        norm = np.linalg.norm(padded)
        if norm > 1e-12:
            padded = padded / norm
        else:
            padded[0] = 1.0
        qml.AmplitudeEmbedding(padded, wires=wires, normalize=True)

    def _variational_circuit(self, params, wires):
        qml.BasicEntanglerLayers(params, wires=wires)

    def _build_kernel_circuit(self):
        dev = self._build_quantum_device()
        n_params = 2 * self.n_qubits

        if self.kernel_type == "angle":

            @qml.qnode(dev)
            def kernel_circuit(x1, x2):
                self._angle_embedding(x1, wires=range(self.n_qubits))
                qml.adjoint(self._angle_embedding)(x2, wires=range(self.n_qubits))
                return qml.probs(wires=range(self.n_qubits))

        elif self.kernel_type == "amplitude":

            @qml.qnode(dev)
            def kernel_circuit(x1, x2):
                self._amplitude_embedding(x1, wires=range(self.n_qubits))
                qml.adjoint(self._amplitude_embedding)(x2, wires=range(self.n_qubits))
                return qml.probs(wires=range(self.n_qubits))

        else:

            @qml.qnode(dev)
            def kernel_circuit(x1, x2):
                return qml.probs(wires=range(self.n_qubits))

        return kernel_circuit

    def _quantum_kernel(self, X1: NDArray, X2: NDArray) -> NDArray:
        if self.kernel_type == "gaussian" or not HAS_PENNYLANE:
            sq_dists = np.sum(X1 ** 2, axis=1, keepdims=True) + \
                       np.sum(X2 ** 2, axis=1) - 2 * (X1 @ X2.T)
            return np.exp(-self.gamma * sq_dists)

        circuit = self._build_kernel_circuit()
        n1, n2 = X1.shape[0], X2.shape[0]
        K = np.zeros((n1, n2))

        batch_size = 50
        for i in range(0, n1, batch_size):
            i_end = min(i + batch_size, n1)
            for j in range(0, n2, batch_size):
                j_end = min(j + batch_size, n2)
                for ii in range(i, i_end):
                    x1 = X1[ii]
                    for jj in range(j, j_end):
                        x2 = X2[jj]
                        try:
                            probs = circuit(x1, x2)
                            K[ii, jj] = probs[0]
                        except Exception:
                            K[ii, jj] = np.exp(
                                -self.gamma * np.sum((x1 - x2) ** 2)
                            )
        return K

    def fit(self, X: NDArray, y: NDArray):
        self.X_fit_ = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        K = self._quantum_kernel(self.X_fit_, self.X_fit_)
        self.svm_ = SVC(kernel="precomputed", C=self.C, probability=True,
                        random_state=self.random_state)
        self.svm_.fit(K, y)
        self.classes_ = np.array([0, 1])
        return self

    def predict(self, X: NDArray) -> NDArray:
        X = np.asarray(X, dtype=np.float64)
        K = self._quantum_kernel(X, self.X_fit_)
        return self.svm_.predict(K)

    def predict_proba(self, X: NDArray) -> NDArray:
        X = np.asarray(X, dtype=np.float64)
        K = self._quantum_kernel(X, self.X_fit_)
        return self.svm_.predict_proba(K)

    def predict_with_uncertainty(self, X: NDArray) -> tuple[NDArray, NDArray, NDArray]:
        proba = self.predict_proba(X)
        anomaly_proba = proba[:, 1]
        predictions = (anomaly_proba >= 0.5).astype(int)
        uncertainty = np.sqrt(anomaly_proba * (1 - anomaly_proba))
        return predictions, anomaly_proba, uncertainty

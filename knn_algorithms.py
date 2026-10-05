import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score

# ============================================================================
# BACKEND SELECTION — controlled by set_backend("gpu") / set_backend("cpu")
# ============================================================================
_BACKEND = "cpu"  # default


def set_backend(backend: str):
    """
    Switch the compute backend.
      "cpu"  — pure NumPy (no PyTorch dependency required)
      "gpu"  — PyTorch with CUDA / MPS / CPU-fallback
    """
    global _BACKEND
    backend = backend.lower()
    if backend not in ("cpu", "gpu"):
        raise ValueError(f"Unknown backend '{backend}'. Use 'cpu' or 'gpu'.")
    if backend == "gpu":
        try:
            import torch  # noqa: F401
        except ImportError:
            raise ImportError(
                "PyTorch is required for the 'gpu' backend. "
                "Install it with: pip install torch"
            )
    _BACKEND = backend


def get_backend() -> str:
    return _BACKEND


# ============================================================================
# NUMPY (CPU) HELPERS
# ============================================================================

def euclidean_distances(X, x):
    return np.linalg.norm(X - x, axis=1)


def min_max_normalize(X):
    denom = X.max(axis=0) - X.min(axis=0)
    denom[denom == 0] = 1
    return (X - X.min(axis=0)) / denom


# ============================================================================
# PYTORCH (GPU) HELPERS
# ============================================================================

def _get_device():
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _to_tensor(X, device):
    import torch
    if isinstance(X, torch.Tensor):
        return X.to(device)
    return torch.tensor(X, dtype=torch.float32, device=device)


# ============================================================================
# SHARED RESOLVE LOGIC
# ============================================================================

def _resolve(y_train, distances, candidate_indices):
    """
    Majority vote over candidate_indices with nearest-neighbor tie-breaking.
    distances: 1-D array of distances from x to every training point.
    candidate_indices: 1-D array of indices into X_train that vote.
    """
    labels = y_train[candidate_indices]
    counter = Counter(labels)
    max_count = counter.most_common(1)[0][1]
    tied = {label for label, count in counter.items() if count == max_count}

    if len(tied) == 1:
        return tied.pop()

    sorted_candidates = candidate_indices[np.argsort(distances[candidate_indices])]
    for idx in sorted_candidates:
        if y_train[idx] in tied:
            return y_train[idx]


# ============================================================================
# CPU IMPLEMENTATIONS
# ============================================================================

class _CustomKNN_CPU:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = np.asarray(X, dtype=float)
        self.y_train = np.asarray(y)
        return self

    def predict(self, X_test):
        X_test = np.asarray(X_test, dtype=float)
        return np.array([self._predict(x) for x in X_test])

    def _predict(self, x):
        distances = euclidean_distances(self.X_train, x)
        k_indices = np.argsort(distances)[:self.k]
        return _resolve(self.y_train, distances, k_indices)

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))

    def get_params(self, deep=True):
        return {"k": self.k}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self


class _MutualKNN_CPU(_CustomKNN_CPU):
    """
    Mutual k-NN (CPU): keeps only neighbors where the relationship is mutual.
    Precomputes k-th neighbor distances at fit time for speed.
    Falls back to standard k-NN when no mutual neighbors exist.
    """

    def fit(self, X, y):
        super().fit(X, y)
        n = len(X)
        self._kth_dist = np.empty(n)
        for i in range(n):
            dists = euclidean_distances(X, X[i])
            sorted_idx = np.argsort(dists)[1:self.k + 1]
            self._kth_dist[i] = dists[sorted_idx[-1]]
        return self

    def _predict(self, x):
        distances = euclidean_distances(self.X_train, x)
        k_indices = np.argsort(distances)[:self.k]

        # Mutual: keep only neighbors for which x would be within their k-NN
        mutual = np.array([idx for idx in k_indices
                           if distances[idx] <= self._kth_dist[idx]])
        candidates = mutual if len(mutual) > 0 else k_indices
        return _resolve(self.y_train, distances, candidates)


class _SymmetricKNN_CPU(_CustomKNN_CPU):
    """
    Symmetric Nearest Neighbor (sNN) rule (CPU).
    Precomputes k-th neighbor distances at fit time.
    Vectorized reverse-neighbor lookup at predict time.
    """

    def fit(self, X, y):
        super().fit(X, y)
        n = len(X)
        self._kth_dist = np.empty(n)
        for i in range(n):
            dists = euclidean_distances(X, X[i])
            sorted_idx = np.argsort(dists)[1:self.k + 1]
            self._kth_dist[i] = dists[sorted_idx[-1]]
        return self

    def _predict(self, x):
        distances = euclidean_distances(self.X_train, x)
        k_indices = np.argsort(distances)[:self.k]

        # Vectorized reverse-neighbor lookup
        reverse = np.where(distances <= self._kth_dist)[0]

        candidates = np.unique(np.concatenate([k_indices, reverse]))
        return _resolve(self.y_train, distances, candidates)


# ============================================================================
# GPU HELPERS — batched operations to avoid OOM on large datasets
# ============================================================================

# Maximum number of float32 elements in a single distance matrix on GPU.
# Controlled by config.yaml → gpu_max_elements.
# Lower this if you hit OOM on your GPU.
_GPU_MAX_ELEMENTS = 32_000_000  # default, overridden by set_gpu_max_elements()


def set_gpu_max_elements(val: int):
    global _GPU_MAX_ELEMENTS
    _GPU_MAX_ELEMENTS = val


def _batched_cdist(A, B, device):
    """
    Compute pairwise Euclidean distances (A, B) → (m, n) in row-batches
    so that no single batch exceeds _GPU_MAX_ELEMENTS floats on the GPU.
    Returns a NumPy array on CPU.
    """
    import torch
    m, n = A.shape[0], B.shape[0]
    batch_rows = max(1, _GPU_MAX_ELEMENTS // n)

    out = np.empty((m, n), dtype=np.float32)
    B_t = _to_tensor(B, device)
    for start in range(0, m, batch_rows):
        end = min(start + batch_rows, m)
        A_batch = _to_tensor(A[start:end], device)
        d = torch.cdist(A_batch, B_t)
        out[start:end] = d.cpu().numpy()
        del A_batch, d
        if device.type == "cuda":
            torch.cuda.empty_cache()
    del B_t
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


def _batched_kth_dist(X_np, k, device):
    """
    For each row i of X, compute the distance to its k-th nearest neighbor
    (excluding self). Returns a 1-D NumPy array of length n.
    """
    import torch
    n = X_np.shape[0]
    batch_rows = max(1, _GPU_MAX_ELEMENTS // n)
    kth = np.empty(n, dtype=np.float32)

    X_t = _to_tensor(X_np, device)
    for start in range(0, n, batch_rows):
        end = min(start + batch_rows, n)
        batch = _to_tensor(X_np[start:end], device)
        d = torch.cdist(batch, X_t)
        # sort each row; index 0 is self (dist≈0), index k is k-th true neighbor
        sorted_d, _ = d.sort(dim=1)
        kth[start:end] = sorted_d[:, k].cpu().numpy()
        del batch, d, sorted_d
        if device.type == "cuda":
            torch.cuda.empty_cache()
    del X_t
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return kth


def _batched_topk(dists_np, k):
    """
    For each row of a (m, n) distance matrix (NumPy), return the indices
    of the k smallest values. Returns (m, k) int array.
    """
    # np.argpartition is O(n) per row — faster than full sort for small k
    if k >= dists_np.shape[1]:
        return np.argsort(dists_np, axis=1)[:, :k]
    idx = np.argpartition(dists_np, k, axis=1)[:, :k]
    # sort the k candidates so idx[:,0] is the nearest
    rows = np.arange(len(dists_np))[:, None]
    order = np.argsort(dists_np[rows, idx], axis=1)
    return idx[rows, order]


# ============================================================================
# GPU IMPLEMENTATIONS
# ============================================================================

class _CustomKNN_GPU:
    def __init__(self, k=3):
        self.k = k
        self.device = _get_device()

    def fit(self, X, y):
        self.X_train = np.asarray(X, dtype=np.float32)
        self.y_train = np.asarray(y)
        return self

    def predict(self, X_test):
        X_test = np.asarray(X_test, dtype=np.float32)
        dists_np = _batched_cdist(X_test, self.X_train, self.device)
        knn_idx = _batched_topk(dists_np, self.k)

        preds = []
        for i in range(len(X_test)):
            preds.append(_resolve(self.y_train, dists_np[i], knn_idx[i]))
        return np.array(preds)

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))

    def get_params(self, deep=True):
        return {"k": self.k}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self


class _MutualKNN_GPU(_CustomKNN_GPU):
    """
    Mutual k-NN (GPU): precomputes training k-th neighbor distances on GPU.
    """

    def fit(self, X, y):
        super().fit(X, y)
        self._train_kth_dist = _batched_kth_dist(self.X_train, self.k, self.device)
        return self

    def predict(self, X_test):
        X_test = np.asarray(X_test, dtype=np.float32)
        dists_np = _batched_cdist(X_test, self.X_train, self.device)
        knn_idx = _batched_topk(dists_np, self.k)

        preds = []
        for i in range(len(X_test)):
            k_indices = knn_idx[i]
            mutual = np.array([idx for idx in k_indices
                               if dists_np[i, idx] <= self._train_kth_dist[idx]])
            candidates = mutual if len(mutual) > 0 else k_indices
            preds.append(_resolve(self.y_train, dists_np[i], candidates))
        return np.array(preds)


class _SymmetricKNN_GPU(_CustomKNN_GPU):
    """
    Symmetric Nearest Neighbor (sNN) rule (GPU).
    Precomputes training k-th neighbor distances on GPU.
    Vectorized reverse-neighbor lookup at predict time.
    """

    def fit(self, X, y):
        super().fit(X, y)
        self._train_kth_dist = _batched_kth_dist(self.X_train, self.k, self.device)
        return self

    def predict(self, X_test):
        X_test = np.asarray(X_test, dtype=np.float32)
        dists_np = _batched_cdist(X_test, self.X_train, self.device)
        knn_idx = _batched_topk(dists_np, self.k)

        preds = []
        for i in range(len(X_test)):
            forward = set(knn_idx[i])
            reverse = set(np.where(dists_np[i] <= self._train_kth_dist)[0])
            candidates = np.array(list(forward | reverse))
            preds.append(_resolve(self.y_train, dists_np[i], candidates))
        return np.array(preds)


# ============================================================================
# PUBLIC API — classes that dispatch to the active backend
# ============================================================================

class CustomKNN:
    """
    Standard k-NN. Backend is selected by set_backend("cpu"|"gpu").
    """
    def __init__(self, k=3):
        self.k = k
        self._impl = None

    def fit(self, X, y):
        if _BACKEND == "gpu":
            self._impl = _CustomKNN_GPU(k=self.k)
        else:
            self._impl = _CustomKNN_CPU(k=self.k)
        self._impl.fit(X, y)
        return self

    def predict(self, X_test):
        return self._impl.predict(X_test)

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))

    def get_params(self, deep=True):
        return {"k": self.k}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        if self._impl is not None:
            self._impl.set_params(**params)
        return self


class MutualKNN:
    """
    Mutual k-NN. Backend is selected by set_backend("cpu"|"gpu").
    """
    def __init__(self, k=3):
        self.k = k
        self._impl = None

    def fit(self, X, y):
        if _BACKEND == "gpu":
            self._impl = _MutualKNN_GPU(k=self.k)
        else:
            self._impl = _MutualKNN_CPU(k=self.k)
        self._impl.fit(X, y)
        return self

    def predict(self, X_test):
        return self._impl.predict(X_test)

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))

    def get_params(self, deep=True):
        return {"k": self.k}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        if self._impl is not None:
            self._impl.set_params(**params)
        return self


class SymmetricKNN:
    """
    Symmetric k-NN (sNN). Backend is selected by set_backend("cpu"|"gpu").
    """
    def __init__(self, k=3):
        self.k = k
        self._impl = None

    def fit(self, X, y):
        if _BACKEND == "gpu":
            self._impl = _SymmetricKNN_GPU(k=self.k)
        else:
            self._impl = _SymmetricKNN_CPU(k=self.k)
        self._impl.fit(X, y)
        return self

    def predict(self, X_test):
        return self._impl.predict(X_test)

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))

    def get_params(self, deep=True):
        return {"k": self.k}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        if self._impl is not None:
            self._impl.set_params(**params)
        return self

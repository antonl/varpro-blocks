from __future__ import print_function
import numpy as np
import arma

def test_shapes():
    a, b = 5, 10
    M = arma.mat(a, b)
    assert all([a == M.n_rows(), b == M.n_cols()]), "incorrect size"

    c = 100
    V = arma.vec(c)
    assert V.n_rows() == c, "incorrect size"

def test_roundtrip_mat():
    a, b = 4, 4
    X = np.random.uniform(0, 1, size=(a, b))
    M = arma.mat(X)

    assert all([a == M.n_rows(), b == M.n_cols()]), "incorrect size"

    rX = M.to_numpy()
    assert np.allclose(X, rX), "data mismatch"

def test_roundtrip_vec():
    c = 100
    v = np.random.uniform(size=(c,))
    aV = arma.vec(v) 
    rV = aV.to_numpy()
    assert np.allclose(rV, v), "data mismatch"


if __name__ == "__main__":
    print("test_shapes: ")
    test_shapes()
    print("test_roundtrip_mat: ")
    test_roundtrip_mat()
    print("test_roundtrip_vec: ")
    test_roundtrip_vec()


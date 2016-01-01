import varpro
import numpy as np

def test_import():
    assert "Hello, world!" == varpro.hello(), "wrong string returned"

def test_arma_vec_create():
    nelem = 20
    x = varpro.arma.Vec(nelem);

    assert nelem == x.n_elem, "wrong number of elements"
    assert nelem == x.n_rows, "wrong number of rows returned"

def test_arma_vec_zero():
    nelem=20
    x = varpro.arma.Vec(nelem)
    x.randn()
    y = np.zeros((20, ), dtype=float)
    assert not np.allclose(x, y), "random not working"
    x.zeros()

    np_x = np.asarray(x)

    assert np.allclose(x, np_x), "incorrect buffer interpretation"
    assert np.allclose(y, x), "incorrect zeroing"

def test_arma_mat_create():
    a,b = 10, 20
    x = varpro.arma.Mat(a, b)
    x.randn()
    y = np.zeros((a, b), dtype=float)
    assert not np.allclose(x, y), "random not working"
    x.zeros()

    np_x = np.asarray(x)

    assert all([x.n_rows == a, x.n_cols == b]), "correct shape of creation"
    assert all([np_x.shape[0] == a, np_x.shape[1] == b]), "buffer shape is wrong"
    assert np_x.flags.f_contiguous, "buffer_info structure is incorrect"

    assert np.allclose(x, np_x), "incorrect buffer interpretation"
    assert np.allclose(y, x), "incorrect zeroing"

def test_arma_mat_roundtrip():
    a, b = 10, 5 
    x = np.random.uniform(size=(a, b))
    y = varpro.arma.Mat(x)
    z = np.asarray(y)

    assert np.allclose(x, z), "roundtrip failed"

def test_arma_vec_roundtrip():
    a = 10  
    x = np.random.uniform(size=(a,))
    y = varpro.arma.Vec(x)
    z = np.asarray(y)

    assert np.allclose(x, z), "roundtrip failed"

if __name__ == "__main__":
    test_import()

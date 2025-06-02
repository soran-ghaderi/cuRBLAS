import cuRBLAS


def test_cuRBLAS():
    assert cuRBLAS.add_one(1) == 2
    assert cuRBLAS.one_plus_one() == 2

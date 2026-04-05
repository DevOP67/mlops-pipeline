def test_import():
    import sklearn

    assert sklearn.__version__ is not None

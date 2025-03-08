import lineworld


def test_recursive_dict_merge():
    """
    Test if the recursive dict merging of toml dicts works as expected
    """
    c1 = {"i": 10, "j": 11}
    c2 = {"i": 12, "k": 13}

    a = {"a": 0, "b": 1, "c": c1, "d": 5}
    b = {"a": -1, "c": c2, "e": 6}

    gt = {"a": -1, "b": 1, "c": {"i": 12, "j": 11, "k": 13}, "d": 5, "e": 6}
    merged = lineworld._recursive_dict_merge(a, b)
    assert gt == merged


def test_apply_config_to_object():
    """
    Test if the recursive dict merging of dicts with dataclass config objects
    (i.e.the FlowlineHatcherConfig) works as expected
    """

    raise NotImplementedError()

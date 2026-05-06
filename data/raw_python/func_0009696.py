def make_filter(**tests):
    """Create a filter from keyword arguments."""
    tests = [AttrTest(k, v) for k, v in tests.items()]
    return Filter(tests)
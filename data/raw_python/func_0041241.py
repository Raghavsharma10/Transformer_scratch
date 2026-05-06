def deepcopy(data):
    """Use pickle to do deep_copy"""
    try:
        return pickle.loads(pickle.dumps(data))
    except TypeError:
        return copy.deepcopy(data)
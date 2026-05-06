def get_pickle_protocol():
    """
    Allow configuration of the pickle protocol on a per-machine basis.
    This way, if you use multiple platforms with different versions of
    pickle, you can configure each of them to use the highest protocol
    supported by all of the machines that you want to be able to
    communicate.
    """
    try:
        protocol_str = os.environ['PYLEARN2_PICKLE_PROTOCOL']
    except KeyError:
        # If not defined, we default to 0 because this is the default
        # protocol used by cPickle.dump (and because it results in
        # maximum portability)
        protocol_str = '0'
    if protocol_str == 'pickle.HIGHEST_PROTOCOL':
        return pickle.HIGHEST_PROTOCOL
    return int(protocol_str)
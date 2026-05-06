def cloudpickle_dumps(obj, dumper=cloudpickle.dumps):
    """ Encode Python objects into a byte stream using cloudpickle. """
    return dumper(obj, protocol=serialization.pickle_protocol)
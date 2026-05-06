def read_proto_object(fobj, klass):
    """Read a block of data and parse using the given protobuf object."""
    log.debug('%s chunk', klass.__name__)
    obj = klass()
    obj.ParseFromString(read_block(fobj))
    log.debug('Header: %s', str(obj))
    return obj
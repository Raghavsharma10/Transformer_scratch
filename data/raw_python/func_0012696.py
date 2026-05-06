def write(parsed_obj, spec=None, filename=None):
    """Writes an object created by `parse` to either a file or a bytearray.

    If the object doesn't end on a byte boundary, zeroes are appended to it
    until it does.
    """
    if not isinstance(parsed_obj, BreadStruct):
        raise ValueError(
            'Object to write must be a structure created '
            'by bread.parse')

    if filename is not None:
        with open(filename, 'wb') as fp:
            parsed_obj._data_bits[:parsed_obj._length].tofile(fp)
    else:
        return bytearray(parsed_obj._data_bits[:parsed_obj._length].tobytes())
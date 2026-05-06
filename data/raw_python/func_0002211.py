def struct_to_dtype(struct):
    """Convert a Structure specification to a numpy structured dtype."""
    # str() around name necessary because protobuf gives unicode names, but dtype doesn't
    # support them on Python 2
    fields = [(str(var.name), data_type_to_numpy(var.dataType, var.unsigned))
              for var in struct.vars]
    for s in struct.structs:
        fields.append((str(s.name), struct_to_dtype(s)))

    log.debug('Structure fields: %s', fields)
    dt = np.dtype(fields)
    return dt
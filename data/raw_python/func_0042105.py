def _write_values(kwargs, variables):
    """Write values of kwargs and return thus-satisfied closures."""
    writeto = []
    for var_name, value in kwargs.items():
        var = variables[var_name]
        var.notify_will_write()
        var.write(value)
        writeto.append(var)
    return _notify_reader_writes(writeto)
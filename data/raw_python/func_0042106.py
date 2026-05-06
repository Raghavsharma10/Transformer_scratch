def _notify_reader_writes(writeto):
    """Notify reader closures about these writes and return a sorted
       list of thus-satisfied closures.
    """
    satisfied = []
    for var in writeto:
        if var.readable:
            for reader in var.readers:
                reader.notify_read_ready()
                if reader.satisfied:
                    satisfied.append(reader)
    return Closure.sort(satisfied)
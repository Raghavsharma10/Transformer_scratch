def getsize(path, bestprefix=True, system=NIST):
    """Return a bitmath instance in the best human-readable representation
of the file size at `path`. Optionally, provide a preferred unit
system by setting `system` to either `bitmath.NIST` (default) or
`bitmath.SI`.

Optionally, set ``bestprefix`` to ``False`` to get ``bitmath.Byte``
instances back.
    """
    _path = os.path.realpath(path)
    size_bytes = os.path.getsize(_path)
    if bestprefix:
        return Byte(size_bytes).best_prefix(system=system)
    else:
        return Byte(size_bytes)
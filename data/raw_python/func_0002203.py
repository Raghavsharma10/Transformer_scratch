def read_ncstream_err(fobj):
    """Handle reading an NcStream error from a file-like object and raise as error."""
    err = read_proto_object(fobj, stream.Error)
    raise RuntimeError(err.message)
def register_binary_type(content_type, dumper, loader):
    """
    Register handling for a binary content type.

    :param str content_type: content type to register the hooks for
    :param dumper: called to decode bytes into a dictionary.
        Calling convention: ``dumper(obj_dict) -> bytes``.
    :param loader: called to encode a dictionary into a byte string.
        Calling convention: ``loader(obj_bytes) -> dict``

    """
    content_type = headers.parse_content_type(content_type)
    content_type.parameters.clear()
    key = str(content_type)
    _content_types[key] = content_type

    handler = _content_handlers.setdefault(key, _ContentHandler(key))
    handler.dict_to_bytes = dumper
    handler.bytes_to_dict = loader
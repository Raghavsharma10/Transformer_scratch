def register_text_type(content_type, default_encoding, dumper, loader):
    """
    Register handling for a text-based content type.

    :param str content_type: content type to register the hooks for
    :param str default_encoding: encoding to use if none is present
        in the request
    :param dumper: called to decode a string into a dictionary.
        Calling convention: ``dumper(obj_dict).encode(encoding) -> bytes``
    :param loader: called to encode a dictionary to a string.
        Calling convention: ``loader(obj_bytes.decode(encoding)) -> dict``

    The decoding of a text content body takes into account decoding
    the binary request body into a string before calling the underlying
    dump/load routines.

    """
    content_type = headers.parse_content_type(content_type)
    content_type.parameters.clear()
    key = str(content_type)
    _content_types[key] = content_type

    handler = _content_handlers.setdefault(key, _ContentHandler(key))
    handler.dict_to_string = dumper
    handler.string_to_dict = loader
    handler.default_encoding = default_encoding or handler.default_encoding
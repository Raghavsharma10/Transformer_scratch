def find_syntax_extension(name):
    """Direct wrapper over cmark_find_syntax_extension."""
    encoded_name = name.encode('utf-8')
    extension = _cmark.lib.cmark_find_syntax_extension(encoded_name)

    if extension == _cmark.ffi.NULL:
        return None
    else:
        return extension
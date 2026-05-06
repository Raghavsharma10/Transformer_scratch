def normalize_uri_path_component(path_component):
    """
    normalize_uri_path_component(path_component) -> str

    Normalize the path component according to RFC 3986.  This performs the
    following operations:
    * Alpha, digit, and the symbols '-', '.', '_', and '~' (unreserved
      characters) are left alone.
    * Characters outside this range are percent-encoded.
    * Percent-encoded values are upper-cased ('%2a' becomes '%2A')
    * Percent-encoded values in the unreserved space (%41-%5A, %61-%7A,
      %30-%39, %2D, %2E, %5F, %7E) are converted to normal characters.

    If a percent encoding is incomplete, the percent is encoded as %25.

    A ValueError exception is thrown if a percent encoding includes non-hex
    characters (e.g. %3z).
    """
    result = BytesIO()

    i = 0
    path_component = path_component.encode("utf-8")
    while i < len(path_component):
        c = indexbytes(path_component, i)
        if c in _rfc3986_unreserved:
            result.write(int2byte(c))
            i += 1
        elif c == _ascii_percent: # percent, '%', 0x25, 37
            if i + 2 >= len(path_component):
                result.write(b"%25")
                i += 1
                continue
            try:
                value = int(path_component[i+1:i+3], 16)
            except ValueError:
                raise ValueError("Invalid %% encoding at position %d" % i)
            
            if value in _rfc3986_unreserved:
                result.write(int2byte(value))
            else:
                result.write(b"%%%02X" % value)
            
            i += 3
        elif c == _ascii_plus:
            # Plus-encoded space.  Convert this to %20.
            result.write(b"%20")
            i += 1
        else:
            result.write(b"%%%02X" % c)
            i += 1
    
    result = result.getvalue()
    if not isinstance(result, string_types):
        result = str(result, "utf-8")
    return result
def smart_bytes(s, encoding="utf-8", strings_only=False, errors="strict"):
    """Return a bytes version of 's' encoded as specified in 'encoding'.

    If strings_only is True, don't convert (some) non-string-like objects.
    """
    if isinstance(s, six.binary_type):
        if encoding == "utf-8":
            return s
        else:
            return s.decode("utf-8", errors).encode(encoding, errors)
    if strings_only and not isinstance(s, six.text_type):
        return s
    if not isinstance(s, six.string_types):
        try:
            if six.PY3:
                return six.text_type(s).encode(encoding)
            else:
                return six.binary_type(s)
        except UnicodeEncodeError:
            return six.text_type(s).encode(encoding, errors)
    else:
        return s.encode(encoding, errors)
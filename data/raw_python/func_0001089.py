def smart_text(s, encoding="utf-8", strings_only=False, errors="strict"):
    """Return a unicode object representing 's'.

    Treats bytes using the 'encoding' codec.

    If strings_only is True, don't convert (some) non-string-like objects.
    """
    if isinstance(s, six.text_type):
        return s
    if strings_only and not isinstance(s, six.string_types):
        return s
    if not isinstance(s, six.string_types):
        if hasattr(s, "__unicode__"):
            s = s.__unicode__()
        else:
            if six.PY3:
                if isinstance(s, six.binary_type):
                    s = six.text_type(s, encoding, errors)
                else:
                    s = six.text_type(s)
            else:
                s = six.text_type(six.binary_type(s), encoding, errors)
    else:
        # Note: We use .decode() here, instead of six.text_type(s, encoding,
        # errors), so that if s is a SafeBytes, it ends up being a
        # SafeText at the end.
        s = s.decode(encoding, errors)
    return s
def forbid_multi_line_headers(name, val):
    """Forbid multi-line headers, to prevent header injection."""
    val = smart_text(val)
    if "\n" in val or "\r" in val:
        raise BadHeaderError(
            "Header values can't contain newlines "
            "(got %r for header %r)" % (val, name)
        )
    try:
        val = val.encode("ascii")
    except UnicodeEncodeError:
        if name.lower() in ("to", "from", "cc"):
            result = []
            for item in val.split(", "):
                nm, addr = parseaddr(item)
                nm = str(Header(nm, DEFAULT_CHARSET))
                result.append(formataddr((nm, str(addr))))
            val = ", ".join(result)
        else:
            val = Header(val, DEFAULT_CHARSET)
    else:
        if name.lower() == "subject":
            val = Header(val)
    return name, val
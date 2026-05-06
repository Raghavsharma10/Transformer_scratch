def _parse_headers(headers):
    "Get headers dict from header string."
    try:
        return dict(line.rstrip().split(": ", 1)
            for line in headers.splitlines()
            if line
        )
    except (TypeError, ValueError) as exc:
        raise SCGIException("Error in SCGI headers %r (%s)" % (headers, exc,))
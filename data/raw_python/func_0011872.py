def _proc_token(spec, mlines):
    """Process line range tokens."""
    spec = spec.strip().replace(" ", "")
    regexp = re.compile(r".*[^0123456789\-,]+.*")
    tokens = spec.split(",")
    cond = any([not item for item in tokens])
    if ("--" in spec) or ("-," in spec) or (",-" in spec) or cond or regexp.match(spec):
        raise RuntimeError("Argument `lrange` is not valid")
    lines = []
    for token in tokens:
        if token.count("-") > 1:
            raise RuntimeError("Argument `lrange` is not valid")
        if "-" in token:
            subtokens = token.split("-")
            lmin, lmax = (
                int(subtokens[0]),
                int(subtokens[1]) if subtokens[1] else mlines,
            )
            for num in range(lmin, lmax + 1):
                lines.append(num)
        else:
            lines.append(int(token))
    if lines != sorted(lines):
        raise RuntimeError("Argument `lrange` is not valid")
    return lines
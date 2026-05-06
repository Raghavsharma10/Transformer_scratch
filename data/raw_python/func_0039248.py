def common_start(*args):
    """ returns the longest common substring from the beginning of sa and sb """
    def _iter():
        for s in zip(*args):
            if len(set(s)) < len(args):
                yield s[0]
            else:
                return

    out = "".join(_iter()).strip()
    result = [s for s in args if not s.startswith(out)]
    result.insert(0, out)
    return ', '.join(result)
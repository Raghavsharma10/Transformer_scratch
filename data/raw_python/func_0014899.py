def slurpChompedLines(file, expand=False):
    r"""Return ``file`` a list of chomped lines. See `slurpLines`."""
    f=_normalizeToFile(file, "r", expand)
    try: return list(chompLines(f))
    finally: f.close()
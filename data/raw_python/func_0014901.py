def spitOut(s, file, binary=False, expand=False):
    r"""Write string `s` into `file` (which can be a string (`str` or
    `unicode`) or a `file` instance)."""
    mode = "w" + ["b",""][not binary]
    file = _normalizeToFile(file, mode=mode, expand=expand)
    try:     file.write(s)
    finally: file.close()
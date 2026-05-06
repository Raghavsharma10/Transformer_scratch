def spitOutLines(lines, file, expand=False):
    r"""Write all the `lines` to `file` (which can be a string/unicode or a
       file handler)."""
    file = _normalizeToFile(file, mode="w", expand=expand)
    try:     file.writelines(lines)
    finally: file.close()
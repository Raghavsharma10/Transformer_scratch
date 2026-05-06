def slurpLines(file, expand=False):
    r"""Read in a complete file (specified by a file handler or a filename
    string/unicode string) as list of lines"""
    file = _normalizeToFile(file, "r", expand)
    try:     return file.readlines()
    finally: file.close()
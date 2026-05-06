def slurp(file, binary=False, expand=False):
    r"""Read in a complete file `file` as a string
    Parameters:

     - `file`: a file handle or a string (`str` or `unicode`).
     - `binary`: whether to read in the file in binary mode (default: False).
    """
    mode = "r" + ["b",""][not binary]
    file = _normalizeToFile(file, mode=mode, expand=expand)
    try: return file.read()
    finally: file.close()
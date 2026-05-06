def head(file_path, lines=10, encoding="utf-8", printed=True,
         errors='strict'):
    """
    Read the first N lines of a file, defaults to 10

    :param file_path: Path to file to read
    :param lines: Number of lines to read in
    :param encoding: defaults to utf-8 to decode as, will fail on binary
    :param printed: Automatically print the lines instead of returning it
    :param errors: Decoding errors: 'strict', 'ignore' or 'replace'
    :return: if printed is false, the lines are returned as a list
    """
    data = []
    with open(file_path, "rb") as f:
        for _ in range(lines):
            try:
                if python_version >= (2, 7):
                    data.append(next(f).decode(encoding, errors=errors))
                else:
                    data.append(next(f).decode(encoding))
            except StopIteration:
                break
    if printed:
        print("".join(data))
    else:
        return data
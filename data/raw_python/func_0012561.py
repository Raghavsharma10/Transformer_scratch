def tail(file_path, lines=10, encoding="utf-8",
         printed=True, errors='strict'):
    """
    A really silly way to get the last N lines, defaults to 10.


    :param file_path: Path to file to read
    :param lines: Number of lines to read in
    :param encoding: defaults to utf-8 to decode as, will fail on binary
    :param printed: Automatically print the lines instead of returning it
    :param errors: Decoding errors: 'strict', 'ignore' or 'replace'
    :return: if printed is false, the lines are returned as a list
    """
    data = deque()

    with open(file_path, "rb") as f:
        for line in f:
            if python_version >= (2, 7):
                data.append(line.decode(encoding, errors=errors))
            else:
                data.append(line.decode(encoding))
            if len(data) > lines:
                data.popleft()
    if printed:
        print("".join(data))
    else:
        return data
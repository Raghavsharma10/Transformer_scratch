def load_data_file(
    file_path,
    file_path_is_relative=False,
    comment_string=DATA_FILE_COMMENT,
    field_separator=DATA_FILE_FIELD_SEPARATOR,
    line_format=None
):
    """
    Load a data file, with one record per line and
    fields separated by ``field_separator``,
    returning a list of tuples.

    It ignores lines starting with ``comment_string`` or empty lines.

    If ``values_per_line`` is not ``None``,
    check that each line (tuple)
    has the prescribed number of values.

    :param str file_path: path of the data file to load
    :param bool file_path_is_relative: if ``True``, ``file_path`` is relative to this source code file
    :param str comment_string: ignore lines starting with this string
    :param str field_separator: fields are separated by this string
    :param str line_format: if not ``None``, parses each line according to the given format
                            (``s`` = string, ``S`` = split string using spaces,
                            ``i`` = int, ``x`` = ignore, ``U`` = Unicode, ``A`` = ASCII)
    :rtype: list of tuples
    """
    raw_tuples = []
    if file_path_is_relative:
        file_path = os.path.join(os.path.dirname(__file__), file_path)
    with io.open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if (len(line) > 0) and (not line.startswith(comment_string)):
                raw_list = line.split(field_separator)
                if len(raw_list) != len(line_format):
                    raise ValueError("Data file '%s' contains a bad line: '%s'" % (file_path, line))
                raw_tuples.append(tuple(raw_list))
    if (line_format is None) or (len(line_format) < 1):
        return raw_tuples
    return [convert_raw_tuple(t, line_format) for t in raw_tuples]
def _get_file_content(source):
    """Return a tuple, each value being a line of the source file.

    Remove empty lines and comments (lines starting with a '#').

    """
    filepath = os.path.join('siglists', source + '.txt')

    lines = []
    with resource_stream(__name__, filepath) as f:
        for i, line in enumerate(f):
            line = line.decode('utf-8', 'strict').strip()
            if not line or line.startswith('#'):
                continue

            try:
                re.compile(line)
            except Exception as ex:
                raise BadRegularExpressionLineError(
                    'Regex error: {} in file {} at line {}'.format(
                        str(ex),
                        filepath,
                        i
                    )
                )

            lines.append(line)

    if source in _SPECIAL_EXTENDED_VALUES:
        lines = lines + _SPECIAL_EXTENDED_VALUES[source]

    return tuple(lines)
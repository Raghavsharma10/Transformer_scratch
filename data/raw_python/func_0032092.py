def expand_includes(text, path='.'):
    """Recursively expands includes in given text."""
    def read_and_expand(match):
        filename = match.group('filename')
        filename = join(path, filename)
        text = read(filename)
        return expand_includes(
            text, path=join(path, dirname(filename)))

    return re.sub(r'^\.\. include:: (?P<filename>.*)$',
                  read_and_expand,
                  text,
                  flags=re.MULTILINE)
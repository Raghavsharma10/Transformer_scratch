def ignore():
    # type: () -> List[str]
    """ Return a list of patterns in the project .gitignore

    Returns:
        list[str]: List of patterns set to be ignored by git.
    """

    def parse_line(line):   # pylint: disable=missing-docstring
        # Decode if necessary
        if not isinstance(line, string_types):
            line = line.decode('utf-8')

        # Strip comment
        line = line.split('#', 1)[0].strip()

        return line

    ignore_files = [
        conf.proj_path('.gitignore'),
        conf.proj_path('.git/info/exclude'),
        config().get('core.excludesfile')
    ]

    result = []
    for ignore_file in ignore_files:
        if not (ignore_file and os.path.exists(ignore_file)):
            continue

        with open(ignore_file) as fp:
            parsed = (parse_line(l) for l in fp.readlines())
            result += [x for x in parsed if x]

    return result
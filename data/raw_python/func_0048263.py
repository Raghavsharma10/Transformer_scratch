def read_version():
    """Read version from the first line starting with digit
    """
    regex = re.compile('^(?P<number>\d.*?) .*$')

    with open('../CHANGELOG.rst') as f:
        for line in f:
            match = regex.match(line)
            if match:
                return match.group('number')
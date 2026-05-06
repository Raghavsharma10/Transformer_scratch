def parse_gdb_version(line):
    r"""Parse the gdb version from the gdb header.

    From GNU coding standards: the version starts after the last space of the
    first line.

    >>> DOCTEST_GDB_VERSIONS = [
    ... r'~"GNU gdb (GDB) 7.5.1\n"',
    ... r'~"GNU gdb (Sourcery CodeBench Lite 2011.09-69) 7.2.50.20100908-cvs\n"',
    ... r'~"GNU gdb (GDB) SUSE (7.5.1-2.5.1)\n"',
    ... r'~"GNU gdb (GDB) Fedora (7.6-32.fc19)\n"',
    ... r'~"GNU gdb (GDB) 7.6.1.dummy\n"',
    ... ]
    >>> for header in DOCTEST_GDB_VERSIONS:
    ...     print(parse_gdb_version(header))
    7.5.1
    7.2.50.20100908
    7.5.1
    7.6
    7.6.1

    """
    if line.startswith('~"') and line.endswith(r'\n"'):
        version = line[2:-3].rsplit(' ', 1)
        if len(version) == 2:
            # Strip after first non digit or '.' character. Allow for linux
            # Suse non conformant implementation that encloses the version in
            # brackets.
            version = ''.join(takewhile(lambda x: x.isdigit() or x == '.',
                                                    version[1].lstrip('(')))
            return version.strip('.')
    return ''
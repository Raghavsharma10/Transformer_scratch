def user_agent(name, version):
    """Return User-Agent ~ "name/version language/version interpreter/version os/version"."""

    def _interpreter():
        name = platform.python_implementation()
        version = platform.python_version()
        bitness = platform.architecture()[0]
        if name == 'PyPy':
            version = '.'.join(map(str, sys.pypy_version_info[:3]))
        full_version = [version]
        if bitness:
            full_version.append(bitness)
        return name, "-".join(full_version)

    tags = [
        (name, version),
        ("python", platform.python_version()),
        _interpreter(),
        ("machine", platform.machine() or 'unknown'),
        ("system", platform.system() or 'unknown'),
        ("platform", platform.platform() or 'unknown'),
    ]

    return ' '.join("{}/{}".format(name, version) for name, version in tags)
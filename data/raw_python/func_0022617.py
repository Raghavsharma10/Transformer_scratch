def get_pkg_version():
    """Get version string by parsing PKG-INFO."""
    try:
        with open("PKG-INFO", "r") as fp:
            rgx = re.compile(r"Version: (\d+)")
            for line in fp.readlines():
                match = rgx.match(line)
                if match:
                    return match.group(1)
    except IOError:
        return None
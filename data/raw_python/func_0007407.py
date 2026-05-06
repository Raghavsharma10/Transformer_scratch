def splitroot(self, part, sep=sep):
        """
        Splits path string into drive, root and relative path

        Uses '/artifactory/' as a splitting point in URI. Everything
        before it, including '/artifactory/' itself is treated as drive.
        The next folder is treated as root, and everything else is taken
        for relative path.
        """
        drv = ''
        root = ''

        base = get_global_base_url(part)
        if base and without_http_prefix(part).startswith(without_http_prefix(base)):
            mark = without_http_prefix(base).rstrip(sep)+sep
            parts = part.split(mark)
        else:
            mark = sep+'artifactory'+sep
            parts = part.split(mark)

        if len(parts) >= 2:
            drv = parts[0] + mark.rstrip(sep)
            rest = sep + mark.join(parts[1:])
        elif part.endswith(mark.rstrip(sep)):
            drv = part
            rest = ''
        else:
            rest = part

        if not rest:
            return drv, '', ''

        if rest == sep:
            return drv, '', ''

        if rest.startswith(sep):
            root, _, part = rest[1:].partition(sep)
            root = sep + root + sep

        return drv, root, part
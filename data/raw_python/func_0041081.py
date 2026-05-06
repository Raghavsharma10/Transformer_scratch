def from_pkg(self):
        """Use pkg_resources to determine the installed package version.
        """
        if self._version is None:
            frame = caller(1)
            pkg = frame.f_globals.get('__package__')
            if pkg is not None:
                self._version = pkg_version(pkg)
        return self
def write(self, version):
        # type: (str) -> None
        """ Write the project version to .py file.

        This will regex search in the file for a
        ``__version__ = VERSION_STRING`` and substitute the version string
        for the new version.
        """
        with open(self.version_file) as fp:
            content = fp.read()

        ver_statement = "__version__ = '{}'".format(version)
        new_content = RE_PY_VERSION.sub(ver_statement, content)
        fs.write_file(self.version_file, new_content)
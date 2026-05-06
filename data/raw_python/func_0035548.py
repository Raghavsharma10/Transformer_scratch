def open(cls, dbfile, encoding=None, fieldnames_lower=True, case_sensitive=True):
        """Context manager. Allows opening a .dbf file.

        .. code-block::

            with Dbf.open('some.dbf') as dbf:
                ...

        :param str|unicode|file dbfile: .dbf filepath or a file-like object.

        :param str|unicode encoding: Encoding used by DB.
            This will be used if there's no encoding information in the DB itself.

        :param bool fieldnames_lower: Lowercase field names.

        :param bool case_sensitive: Whether DB filename is case sensitive.

        :rtype: Dbf
        """
        if not case_sensitive:
            if isinstance(dbfile, string_types):
                dbfile = pick_name(dbfile, listdir(path.dirname(dbfile)))

        with open(dbfile, 'rb') as f:
            yield cls(f, encoding=encoding, fieldnames_lower=fieldnames_lower)
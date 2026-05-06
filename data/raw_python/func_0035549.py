def open_zip(cls, dbname, zipped, encoding=None, fieldnames_lower=True, case_sensitive=True):
        """Context manager. Allows opening a .dbf file from zip archive.

        .. code-block::

            with Dbf.open_zip('some.dbf', 'myarch.zip') as dbf:
                ...

        :param str|unicode dbname: .dbf file name

        :param str|unicode|file zipped: .zip file path or a file-like object.

        :param str|unicode encoding: Encoding used by DB.
            This will be used if there's no encoding information in the DB itself.

        :param bool fieldnames_lower: Lowercase field names.

        :param bool case_sensitive: Whether DB filename is case sensitive.

        :rtype: Dbf
        """
        with ZipFile(zipped, 'r') as zip_:

            if not case_sensitive:
                dbname = pick_name(dbname, zip_.namelist())

            with zip_.open(dbname) as f:
                yield cls(f, encoding=encoding, fieldnames_lower=fieldnames_lower)
def open_db(db, zipped=None, encoding=None, fieldnames_lower=True, case_sensitive=True):
    """Context manager. Allows reading DBF file (maybe even from zip).

    :param str|unicode|file db: .dbf file name or a file-like object.

    :param str|unicode zipped: .zip file path or a file-like object.

    :param str|unicode encoding: Encoding used by DB.
        This will be used if there's no encoding information in the DB itself.

    :param bool fieldnames_lower: Lowercase field names.

    :param bool case_sensitive: Whether DB filename is case sensitive.

    :rtype: Dbf
    """
    kwargs = dict(
        encoding=encoding,
        fieldnames_lower=fieldnames_lower,
        case_sensitive=case_sensitive,
    )

    if zipped:
        with Dbf.open_zip(db, zipped, **kwargs) as dbf:
            yield dbf

    else:
        with Dbf.open(db, **kwargs) as dbf:
            yield dbf
def file_from_content(content):
    """Provides a real file object from file content

    Converts ``FileStorage``, ``FileIntent`` and
    ``bytes`` to an actual file.
    """
    f = content
    if isinstance(content, cgi.FieldStorage):
        f = content.file
    elif isinstance(content, FileIntent):
        f = content._fileobj
    elif isinstance(content, byte_string):
        f = SpooledTemporaryFile(INMEMORY_FILESIZE)
        f.write(content)
    f.seek(0)
    return f
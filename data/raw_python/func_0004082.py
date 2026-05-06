def safe_filename(file_name, sep='_', default=None, extension=None):
    """Create a secure filename for plain file system storage."""
    if file_name is None:
        return decode_path(default)

    file_name = decode_path(file_name)
    file_name = os.path.basename(file_name)
    file_name, _extension = os.path.splitext(file_name)
    file_name = _safe_name(file_name, sep=sep)
    if file_name is None:
        return decode_path(default)
    file_name = file_name[:MAX_LENGTH]
    extension = _safe_name(extension or _extension, sep=sep)
    if extension is not None:
        file_name = '.'.join((file_name, extension))
        file_name = file_name[:MAX_LENGTH]
    return file_name
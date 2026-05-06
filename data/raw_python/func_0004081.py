def _safe_name(file_name, sep):
    """Convert the file name to ASCII and normalize the string."""
    file_name = stringify(file_name)
    if file_name is None:
        return
    file_name = ascii_text(file_name)
    file_name = category_replace(file_name, UNICODE_CATEGORIES)
    file_name = collapse_spaces(file_name)
    if file_name is None or not len(file_name):
        return
    return file_name.replace(WS, sep)
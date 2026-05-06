def has_lexer(source_path):
    """
    Initial quick check if there is a lexer for ``source_path``. This removes
    the need for calling :py:func:`pygments.lexers.guess_lexer_for_filename()`
    which fully reads the source file.
    """
    result = bool(pygments.lexers.find_lexer_class_for_filename(source_path))
    if not result:
        suffix = os.path.splitext(os.path.basename(source_path))[1].lstrip('.')
        result = suffix in _SUFFIX_TO_FALLBACK_LEXER_MAP
    return result
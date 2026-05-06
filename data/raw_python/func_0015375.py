def _map_xpath_flags_to_re(expr: str, xpath_flags: str) -> Tuple[int, str]:
    """ Map `5.6.2 Flags <https://www.w3.org/TR/xpath-functions-31/#flags>`_  to python

    :param expr: match pattern
    :param xpath_flags: xpath flags
    :returns: python flags / modified match pattern
    """
    python_flags: int = 0
    modified_expr = expr
    if xpath_flags is None:
        xpath_flags = ""

    if 's' in xpath_flags:
        python_flags |= re.DOTALL
    if 'm' in xpath_flags:
        python_flags |= re.MULTILINE
    if 'i' in xpath_flags:
        python_flags |= re.IGNORECASE
    if 'x' in xpath_flags:
        modified_expr = re.sub(r'[\t\n\r ]|\[[^\]]*\]', _char_class_escape, modified_expr)
    if 'q' in xpath_flags:
        modified_expr = re.escape(modified_expr)

    return python_flags, modified_expr
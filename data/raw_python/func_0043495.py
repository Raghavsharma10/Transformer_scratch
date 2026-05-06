def sanitize_latex(string):
    """
    Sanitize a string for input to LaTeX.

    Replacements taken from `Stack Overflow
    <http://stackoverflow.com/questions/2627135/how-do-i-sanitize-latex-input>`_

    **Parameters**

    string: str

    **Returns**

    sanitized_string: str
    """
    sanitized_string = string
    for old, new in _latex_replacements:
        sanitized_string = sanitized_string.replace(old, new)
    return sanitized_string
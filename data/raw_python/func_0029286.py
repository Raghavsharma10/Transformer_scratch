def strip_html(string, keep_tag_content=False):
    """
    Remove html code contained into the given string.

    :param string: String to manipulate.
    :type string: str
    :param keep_tag_content: True to preserve tag content, False to remove tag and its content too (default).
    :type keep_tag_content: bool
    :return: String with html removed.
    :rtype: str
    """
    r = HTML_TAG_ONLY_RE if keep_tag_content else HTML_RE
    return r.sub('', string)
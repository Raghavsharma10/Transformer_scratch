def _xml_tag_filter(s: str, strip_namespaces: bool) -> str:
    """
    Returns tag name and optionally strips namespaces.
    :param el: Element
    :param strip_namespaces: Strip namespace prefix
    :return: str
    """
    if strip_namespaces:
        ns_end = s.find('}')
        if ns_end != -1:
            s = s[ns_end+1:]
        else:
            ns_end = s.find(':')
            if ns_end != -1:
                s = s[ns_end+1:]
    return s
def prettystr(subtree):
    """Print an element tree with nice indentation.

    Prettyprinting a whole VOEvent often doesn't seem to work, probably for
    issues relating to whitespace cf.
    http://lxml.de/FAQ.html#why-doesn-t-the-pretty-print-option-reformat-my-xml-output
    This function is a quick workaround for prettyprinting a subsection
    of a VOEvent, for easier desk-checking.

    Args:
        subtree(:class`lxml.etree.ElementTree`): A node in the VOEvent element tree.
    Returns:
        str: Prettyprinted string representation of the raw XML.
    """
    subtree = deepcopy(subtree)
    lxml.objectify.deannotate(subtree)
    lxml.etree.cleanup_namespaces(subtree)
    return lxml.etree.tostring(subtree, pretty_print=True).decode(
        encoding="utf-8")
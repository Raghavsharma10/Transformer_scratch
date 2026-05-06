def etree_to_string(root, pretty_print=True, xml_declaration=True,
                    encoding='utf-8'):
    """Dump XML etree as a string."""
    return etree.tostring(
        root,
        pretty_print=pretty_print,
        xml_declaration=xml_declaration,
        encoding=encoding,
    ).decode('utf-8')
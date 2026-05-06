def get_parsed_content(metadata_content):
    """
    Parses any of the following types of content:
    1. XML string or file object: parses XML content
    2. MetadataParser instance: deep copies xml_tree
    3. Dictionary with nested objects containing:
        - name (required): the name of the element tag
        - text: the text contained by element
        - tail: text immediately following the element
        - attributes: a Dictionary containing element attributes
        - children: a List of converted child elements

    :raises InvalidContent: if the XML is invalid or does not conform to a supported metadata standard
    :raises NoContent: If the content passed in is null or otherwise empty

    :return: the XML root along with an XML Tree parsed by and compatible with element_utils
    """

    _import_parsers()  # Prevents circular dependencies between modules

    xml_tree = None

    if metadata_content is None:
        raise NoContent('Metadata has no data')
    else:
        if isinstance(metadata_content, MetadataParser):
            xml_tree = deepcopy(metadata_content._xml_tree)
        elif isinstance(metadata_content, dict):
            xml_tree = get_element_tree(metadata_content)
        else:
            try:
                # Strip name spaces from file or XML content
                xml_tree = get_element_tree(metadata_content)
            except Exception:
                xml_tree = None  # Several exceptions possible, outcome is the same

    if xml_tree is None:
        raise InvalidContent(
            'Cannot instantiate a {parser_type} parser with invalid content to parse',
            parser_type=type(metadata_content).__name__
        )

    xml_root = get_element_name(xml_tree)

    if xml_root is None:
        raise NoContent('Metadata contains no data')
    elif xml_root not in VALID_ROOTS:
        content = type(metadata_content).__name__
        raise InvalidContent('Invalid root element for {content}: {xml_root}', content=content, xml_root=xml_root)

    return xml_root, xml_tree
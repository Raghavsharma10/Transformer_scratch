def xml_to_dict(xml_bytes: bytes, tags: list=[], array_tags: list=[], int_tags: list=[],
                strip_namespaces: bool=True, parse_attributes: bool=True,
                value_key: str='@', attribute_prefix: str='@',
                document_tag: bool=False) -> dict:
    """
    Parses XML string to dict. In case of simple elements (no children, no attributes) value is stored as is.
    For complex elements value is stored in key '@', attributes '@xxx' and children as sub-dicts.
    Optionally strips namespaces.

    For example:
        <Doc version="1.2">
          <A class="x">
            <B class="x2">hello</B>
          </A>
          <A class="y">
            <B class="y2">world</B>
          </A>
          <C>value node</C>
        </Doc>
    is returned as follows:
        {'@version': '1.2',
         'A': [{'@class': 'x', 'B': {'@': 'hello', '@class': 'x2'}},
               {'@class': 'y', 'B': {'@': 'world', '@class': 'y2'}}],
         'C': 'value node'}

    Args:
        xml_bytes: XML file contents in bytes
        tags: list of tags to parse (pass empty to return all chilren of top-level tag)
        array_tags: list of tags that should be treated as arrays by default
        int_tags: list of tags that should be treated as ints
        strip_namespaces: if true namespaces will be stripped
        parse_attributes: Elements with attributes are stored as complex types with '@' identifying text value and @xxx identifying each attribute
        value_key: Key to store (complex) element value. Default is '@'
        attribute_prefix: Key prefix to store element attribute values. Default is '@'
        document_tag: Set True if Document root tag should be included as well

    Returns: dict
    """
    from xml.etree import ElementTree as ET

    root = ET.fromstring(xml_bytes)
    if tags:
        if document_tag:
            raise Exception('xml_to_dict: document_tag=True does not make sense when using selective tag list since selective tag list finds tags from the whole document, not only directly under root document tag')
        root_elements = []
        for tag in tags:
            root_elements.extend(root.iter(tag))
    else:
        root_elements = list(root)

    data = {}
    for el in root_elements:
        _xml_set_element_data_r(data, el, array_tags=array_tags, int_tags=int_tags,
                                strip_namespaces=strip_namespaces, parse_attributes=parse_attributes,
                                value_key=value_key, attribute_prefix=attribute_prefix)

    # set root attributes
    if parse_attributes:
        for a_key, a_val in root.attrib.items():
            data[attribute_prefix + _xml_tag_filter(a_key, strip_namespaces)] = a_val

    return data if not document_tag else {root.tag: data}
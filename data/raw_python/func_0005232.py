def dict_to_element(doc: dict, value_key: str='@', attribute_prefix: str='@') -> Element:
    """
    Generates XML Element from dict.
    Generates complex elements by assuming element attributes are prefixed with '@', and value is stored to plain '@'
    in case of complex element. Children are sub-dicts.

    For example:
        {
            'Doc': {
                '@version': '1.2',
                'A': [{'@class': 'x', 'B': {'@': 'hello', '@class': 'x2'}},
                      {'@class': 'y', 'B': {'@': 'world', '@class': 'y2'}}],
                'C': 'value node',
            }
         }
    is returned as follows:
        <?xml version="1.0" ?>
        <Doc version="1.2">
            <A class="x">
                <B class="x2">hello</B>
            </A>
            <A class="y">
                <B class="y2">world</B>
            </A>
            <C>value node</C>
        </Doc>

    Args:
        doc: dict. Must have sigle root key dict.
        value_key: Key to store (complex) element value. Default is '@'
        attribute_prefix: Key prefix to store element attribute values. Default is '@'

    Returns: xml.etree.ElementTree.Element
    """
    from xml.etree import ElementTree as ET

    if len(doc) != 1:
        raise Exception('Invalid data dict for XML generation, document root must have single element')

    for tag, data in doc.items():
        el = ET.Element(tag)
        assert isinstance(el, Element)
        _xml_element_set_data_r(el, data, value_key, attribute_prefix)
        return el
def make_indicatoritem_node(condition,
                            document,
                            search,
                            content_type,
                            content,
                            preserve_case=False,
                            negate=False,
                            context_type='mir',
                            nid=None):
    """
    This makes a IndicatorItem element.  This contains the actual threat intelligence in the IOC.

    :param condition: This is the condition of the item ('is', 'contains', 'matches', etc). The following contants in ioc_api may be used:
==================== =====================================================
Constant             Meaning
==================== =====================================================
ioc_api.IS           Exact String match.
ioc_api.CONTAINS     Substring match.
ioc_api.MATCHES      Regex match.
ioc_api.STARTS_WITH  String match at the beginning of a string.
ioc_api.ENDS_WITH    String match at the end of a string.
ioc_api.GREATER_THAN Integer match indicating a greater than (>) operation.
ioc_api.LESS_THAN    Integer match indicator a less than (<) operation.
==================== =====================================================

    :param document: Denotes the type of document to look for the encoded artifact in.
    :param search: Specifies what attribute of the document type the encoded value is.
    :param content_type: This is the display type of the item. This is normally derived from the iocterm for the search value.
    :param content: The threat intelligence that is being encoded.
    :param preserve_case: Specifiy that the content should be treated in a case sensitive manner.
    :param negate: Specifify that the condition is negated. An example of this is:
       @condition = 'is' & @negate = 'true' would be equal to the
       @condition = 'isnot' in OpenIOC 1.0.
    :param context_type: Gives context to the document/search information.
    :param nid: This is used to provide a GUID for the IndicatorItem. The ID should NOT be specified under normal
     circumstances.
    :return: an elementTree Element item
    """
    # validate condition
    if condition not in VALID_INDICATORITEM_CONDITIONS:
        raise ValueError('Invalid IndicatorItem condition [{}]'.format(condition))
    ii_node = et.Element('IndicatorItem')
    if nid:
        ii_node.attrib['id'] = nid
    else:
        ii_node.attrib['id'] = ioc_et.get_guid()
    ii_node.attrib['condition'] = condition
    if preserve_case:
        ii_node.attrib['preserve-case'] = 'true'
    else:
        ii_node.attrib['preserve-case'] = 'false'
    if negate:
        ii_node.attrib['negate'] = 'true'
    else:
        ii_node.attrib['negate'] = 'false'
    context_node = ioc_et.make_context_node(document, search, context_type)
    content_node = ioc_et.make_content_node(content_type, content)
    ii_node.append(context_node)
    ii_node.append(content_node)
    return ii_node
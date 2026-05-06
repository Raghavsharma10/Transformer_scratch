def make_indicator_node(operator, nid=None):
    """
    This makes a Indicator node element.  These allow the construction of a logic tree within the IOC.

    :param operator: String 'AND' or 'OR'.  The constants ioc_api.OR and ioc_api.AND may be used as well.
    :param nid: This is used to provide a GUID for the Indicator. The ID should NOT be specified under normal circumstances.
    :return: elementTree element
    """
    if operator.upper() not in VALID_INDICATOR_OPERATORS:
        raise ValueError('Indicator operator must be in [{}].'.format(VALID_INDICATOR_OPERATORS))
    i_node = et.Element('Indicator')
    if nid:
        i_node.attrib['id'] = nid
    else:
        i_node.attrib['id'] = ioc_et.get_guid()
    i_node.attrib['operator'] = operator.upper()
    return i_node
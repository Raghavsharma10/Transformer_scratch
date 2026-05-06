def get_top_level_indicator_node(root_node):
    """
    This returns the first top level Indicator node under the criteria node.

    :param root_node: Root node of an etree.
    :return: an elementTree Element item, or None if no item is found.
    """
    if root_node.tag != 'OpenIOC':
        raise IOCParseError('Root tag is not "OpenIOC" [{}].'.format(root_node.tag))
    elems = root_node.xpath('criteria/Indicator')
    if len(elems) == 0:
        log.warning('No top level Indicator node found.')
        return None
    elif len(elems) > 1:
        log.warning('Multiple top level Indicator nodes found.  This is not a valid MIR IOC.')
        return None
    else:
        top_level_indicator_node = elems[0]
    if top_level_indicator_node.get('operator').lower() != 'or':
        log.warning('Top level Indicator/@operator attribute is not "OR".  This is not a valid MIR IOC.')
    return top_level_indicator_node
def write_ioc_string(root, force=False):
    """
    Serialize an IOC, as defined by a set of etree Elements, to a String.
    :param root: etree Element to serialize.  Should have the tag 'OpenIOC'
    :param force: Skip the root node tag check.
    :return:
    """
    root_tag = 'OpenIOC'
    if not force and root.tag != root_tag:
        raise ValueError('Root tag is not "{}".'.format(root_tag))
    default_encoding = 'utf-8'
    tree = root.getroottree()
    # noinspection PyBroadException
    try:
        encoding = tree.docinfo.encoding
    except:
        log.debug('Failed to get encoding from docinfo')
        encoding = default_encoding
    return et.tostring(tree, encoding=encoding, xml_declaration=True, pretty_print=True)
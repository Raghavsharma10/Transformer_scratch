def write_ioc(root, output_dir=None, force=False):
    """
    Serialize an IOC, as defined by a set of etree Elements, to a .IOC file.

    :param root: etree Element to write out.  Should have the tag 'OpenIOC'
    :param output_dir: Directory to write the ioc out to.  default is current working directory.
    :param force: If set, skip the root node tag check.
    :return: True, unless an error occurs while writing the IOC.
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
    ioc_id = root.attrib['id']
    fn = ioc_id + '.ioc'
    if output_dir:
        fn = os.path.join(output_dir, fn)
    else:
        fn = os.path.join(os.getcwd(), fn)
    try:
        with open(fn, 'wb') as fout:
            fout.write(et.tostring(tree, encoding=encoding, xml_declaration=True, pretty_print=True))
    except (IOError, OSError):
        log.exception('Failed to write out IOC')
        return False
    except:
        raise
    return True
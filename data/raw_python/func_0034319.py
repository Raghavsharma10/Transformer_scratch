def to_xml(node, pretty=False):
    """ convert an etree node to xml """
    fout = Sio()
    etree = et.ElementTree(node)

    etree.write(fout)
    xml = fout.getvalue()
    if pretty:
        xml = pretty_xml(xml, True)
    return xml
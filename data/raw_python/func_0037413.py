def dump(voevent, file, pretty_print=True, xml_declaration=True):
    """Writes the voevent to the file object.

    e.g.::

        with open('/tmp/myvoevent.xml','wb') as f:
            voeventparse.dump(v, f)

    Args:
        voevent(:class:`Voevent`): Root node of the VOevent etree.
        file (io.IOBase): An open (binary mode) file object for writing.
        pretty_print
        pretty_print(bool): See :func:`dumps`
        xml_declaration(bool): See :func:`dumps`
    """
    file.write(dumps(voevent, pretty_print, xml_declaration))
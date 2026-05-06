def response(args):
    """Response endpoint."""
    e_tree = getattr(xml, args['verb'].lower())(**args)

    response = make_response(etree.tostring(
        e_tree,
        pretty_print=True,
        xml_declaration=True,
        encoding='UTF-8',
    ))
    response.headers['Content-Type'] = 'text/xml'
    return response
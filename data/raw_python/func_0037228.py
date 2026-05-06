def _build_from(baseip):
    """ Build URL for description.xml from ip """
    from ipaddress import ip_address
    try:
        ip_address(baseip)
    except ValueError:
        # """attempt to construct url but the ip format has changed"""
        # logger.warning("Format of internalipaddress changed: %s", baseip)
        if 'http' not in baseip[0:4].lower():
            baseip = urlunsplit(['http', baseip, '', '', ''])
        spl = urlsplit(baseip)
        if '.xml' not in spl.path:
            sep = '' if spl.path.endswith('/') else '/'
            spl = spl._replace(path=spl.path+sep+'description.xml')
        return spl.geturl()
    else:
        # construct url knowing baseip is a pure ip
        return  urlunsplit(('http', baseip, '/description.xml', '', ''))
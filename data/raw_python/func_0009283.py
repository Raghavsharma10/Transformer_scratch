def get_bewit(resource):
    """
    Returns a bewit identifier for the resource as a string.

    :param resource:
        Resource to generate a bewit for
    :type resource: `mohawk.base.Resource`
    """
    if resource.method != 'GET':
        raise ValueError('bewits can only be generated for GET requests')
    if resource.nonce != '':
        raise ValueError('bewits must use an empty nonce')
    mac = calculate_mac(
        'bewit',
        resource,
        None,
    )

    if isinstance(mac, six.binary_type):
        mac = mac.decode('ascii')

    if resource.ext is None:
        ext = ''
    else:
        validate_header_attr(resource.ext, name='ext')
        ext = resource.ext

    # b64encode works only with bytes in python3, but all of our parameters are
    # in unicode, so we need to encode them. The cleanest way to do this that
    # works in both python 2 and 3 is to use string formatting to get a
    # unicode string, and then explicitly encode it to bytes.
    inner_bewit = u"{id}\\{exp}\\{mac}\\{ext}".format(
        id=resource.credentials['id'],
        exp=resource.timestamp,
        mac=mac,
        ext=ext,
    )
    inner_bewit_bytes = inner_bewit.encode('ascii')
    bewit_bytes = urlsafe_b64encode(inner_bewit_bytes)
    # Now decode the resulting bytes back to a unicode string
    return bewit_bytes.decode('ascii')
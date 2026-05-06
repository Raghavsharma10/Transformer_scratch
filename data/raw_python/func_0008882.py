def armor(data, versioned=True):
    """
    Returns a string in ASCII Armor format, for the given binary data. The
    output of this is compatiple with pgcrypto's armor/dearmor functions.
    """
    template = '-----BEGIN PGP MESSAGE-----\n%(headers)s%(body)s\n=%(crc)s\n-----END PGP MESSAGE-----'
    body = base64.b64encode(data)
    # The 24-bit CRC should be in big-endian, strip off the first byte (it's already masked in crc24).
    crc = base64.b64encode(struct.pack('>L', crc24(data))[1:])
    return template % {
        'headers': 'Version: django-pgcrypto %s\n\n' % __version__ if versioned else '\n',
        'body': body.decode('ascii'),
        'crc': crc.decode('ascii'),
    }
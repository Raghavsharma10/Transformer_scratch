def assert_fingerprint(cert, fingerprint):
    """
    Checks if given fingerprint matches the supplied certificate.

    :param cert:
        Certificate as bytes object.
    :param fingerprint:
        Fingerprint as string of hexdigits, can be interspersed by colons.
    """

    # Maps the length of a digest to a possible hash function producing
    # this digest.
    hashfunc_map = {
        16: md5,
        20: sha1,
        32: sha256,
    }

    fingerprint = fingerprint.replace(':', '').lower()
    digest_length, odd = divmod(len(fingerprint), 2)

    if odd or digest_length not in hashfunc_map:
        raise SSLError('Fingerprint is of invalid length.')

    # We need encode() here for py32; works on py2 and p33.
    fingerprint_bytes = unhexlify(fingerprint.encode())

    hashfunc = hashfunc_map[digest_length]

    cert_digest = hashfunc(cert).digest()

    if not cert_digest == fingerprint_bytes:
        raise SSLError('Fingerprints did not match. Expected "{0}", got "{1}".'
                       .format(hexlify(fingerprint_bytes),
                               hexlify(cert_digest)))
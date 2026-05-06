def csr_for_names(names, key):
    """
    Generate a certificate signing request for the given names and private key.

    ..  seealso:: `acme.client.Client.request_issuance`

    ..  seealso:: `generate_private_key`

    :param ``List[str]``: One or more names (subjectAltName) for which to
        request a certificate.
    :param key: A Cryptography private key object.

    :rtype: `cryptography.x509.CertificateSigningRequest`
    :return: The certificate request message.
    """
    if len(names) == 0:
        raise ValueError('Must have at least one name')
    if len(names[0]) > 64:
        common_name = u'san.too.long.invalid'
    else:
        common_name = names[0]
    return (
        x509.CertificateSigningRequestBuilder()
        .subject_name(x509.Name([
            x509.NameAttribute(NameOID.COMMON_NAME, common_name)]))
        .add_extension(
            x509.SubjectAlternativeName(list(map(x509.DNSName, names))),
            critical=False)
        .sign(key, hashes.SHA256(), default_backend()))
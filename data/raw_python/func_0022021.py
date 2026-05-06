def decode_csr(b64der):
    """
    Decode JOSE Base-64 DER-encoded CSR.

    :param str b64der: The encoded CSR.

    :rtype: `cryptography.x509.CertificateSigningRequest`
    :return: The decoded CSR.
    """
    try:
        return x509.load_der_x509_csr(
            decode_b64jose(b64der), default_backend())
    except ValueError as error:
        raise DeserializationError(error)
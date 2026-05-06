def generate_tls_sni_01_cert(server_name, key_type=u'rsa',
                             _generate_private_key=None):
    """
    Generate a certificate/key pair for responding to a tls-sni-01 challenge.

    :param str server_name: The SAN the certificate should have.
    :param str key_type: The type of key to generate; usually not necessary.

    :rtype: ``Tuple[`~cryptography.x509.Certificate`, PrivateKey]``
    :return: A tuple of the certificate and private key.
    """
    key = (_generate_private_key or generate_private_key)(key_type)
    name = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, u'acme.invalid')])
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .not_valid_before(datetime.now() - timedelta(seconds=3600))
        .not_valid_after(datetime.now() + timedelta(seconds=3600))
        .serial_number(int(uuid.uuid4()))
        .public_key(key.public_key())
        .add_extension(
            x509.SubjectAlternativeName([x509.DNSName(server_name)]),
            critical=False)
        .sign(
            private_key=key,
            algorithm=hashes.SHA256(),
            backend=default_backend())
        )
    return (cert, key)
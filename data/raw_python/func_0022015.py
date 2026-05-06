def load_or_create_client_key(pem_path):
    """
    Load the client key from a directory, creating it if it does not exist.

    .. note:: The client key that will be created will be a 2048-bit RSA key.

    :type pem_path: ``twisted.python.filepath.FilePath``
    :param pem_path: The certificate directory
        to use, as with the endpoint.
    """
    acme_key_file = pem_path.asTextMode().child(u'client.key')
    if acme_key_file.exists():
        key = serialization.load_pem_private_key(
            acme_key_file.getContent(),
            password=None,
            backend=default_backend())
    else:
        key = generate_private_key(u'rsa')
        acme_key_file.setContent(
            key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.NoEncryption()))
    return JWKRSA(key=key)
def _generate():
    """
    Generate a new SSH key pair.
    """
    privateKey = rsa.generate_private_key(
        public_exponent=65537,
        key_size=4096,
        backend=default_backend())
    return Key(privateKey).toString('openssh')
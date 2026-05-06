def salt_hash_from_plaintext(password):
    """
    Create a XOR encrypted PBKDF2 salted checksum from a plaintext password.

    >>> seed_generator.DEBUG=True # Generate always the same seed for tests

    >>> salt, data = salt_hash_from_plaintext("test")
    >>> salt == 'DEBUG'
    True
    >>> data =='pbkdf2_sha1$5$DEBUG$a2220ab7dea891f260edd481$50530c0e530f030b08070353'
    True
    """
    init_pbkdf2_salt = seed_generator(app_settings.PBKDF2_SALT_LENGTH)
    pbkdf2_temp_hash = hexlify_pbkdf2(
        password,
        salt=init_pbkdf2_salt,
        iterations=app_settings.ITERATIONS1,
        length=app_settings.PBKDF2_BYTE_LENGTH
    )

    first_pbkdf2_part = pbkdf2_temp_hash[:PBKDF2_HALF_HEX_LENGTH]
    second_pbkdf2_part = pbkdf2_temp_hash[PBKDF2_HALF_HEX_LENGTH:]

    encrypted_part = xor_crypt.encrypt(first_pbkdf2_part, key=second_pbkdf2_part)

    # log.debug("locals():\n%s", pprint.pformat(locals()))
    return init_pbkdf2_salt, encrypted_part
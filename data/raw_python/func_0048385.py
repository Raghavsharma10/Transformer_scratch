def _simulate_client(plaintext_password, init_pbkdf2_salt, cnonce, server_challenge):
    """
    A implementation of the JavaScript client part.
    Needful for finding bugs.
    """
    # log.debug("_simulate_client(plaintext_password='%s', init_pbkdf2_salt='%s', cnonce='%s', server_challenge='%s')",
    #     plaintext_password, init_pbkdf2_salt, cnonce, server_challenge
    # )
    pbkdf2_temp_hash = hexlify_pbkdf2(
        plaintext_password,
        salt=init_pbkdf2_salt,
        iterations=app_settings.ITERATIONS1,
        length=app_settings.PBKDF2_BYTE_LENGTH
    )
    first_pbkdf2_part = pbkdf2_temp_hash[:PBKDF2_HALF_HEX_LENGTH]
    second_pbkdf2_part = pbkdf2_temp_hash[PBKDF2_HALF_HEX_LENGTH:]

    second_pbkdf2_salt = cnonce + server_challenge
    pbkdf2_hash = hexlify_pbkdf2(
        first_pbkdf2_part,
        salt=second_pbkdf2_salt,
        iterations=app_settings.ITERATIONS2,
        length=app_settings.PBKDF2_BYTE_LENGTH
    )
    # log.debug("_simulate_client() locals():\n%s", pprint.pformat(locals()))
    return pbkdf2_hash, second_pbkdf2_part
def check_secure_js_login(secure_password, encrypted_part, server_challenge):
    """
    first_pbkdf2_part = xor_decrypt(encrypted_part, key=second_pbkdf2_part)
    test_hash = pbkdf2(first_pbkdf2_part, key=cnonce + server_challenge)
    compare test_hash with transmitted pbkdf2_hash
    """
    # log.debug("check_secure_js_login(secure_password='%s', encrypted_part='%s', server_challenge='%s')",
    #     secure_password, encrypted_part, server_challenge
    # )

    pbkdf2_hash, second_pbkdf2_part, cnonce = split_secure_password(secure_password)
    # log.debug("split_secure_password(): pbkdf2_hash='%s', second_pbkdf2_part='%s', cnonce='%s'",
    #     pbkdf2_hash, second_pbkdf2_part, cnonce
    # )

    first_pbkdf2_part = xor_crypt.decrypt(encrypted_part, key=second_pbkdf2_part)

    test_hash = hexlify_pbkdf2(
        first_pbkdf2_part,
        cnonce + server_challenge,
        iterations=app_settings.ITERATIONS2,
        length=app_settings.PBKDF2_BYTE_LENGTH
    )
    # log.debug("check_secure_js_login() locals():\n%s", pprint.pformat(locals()))
    if test_hash != pbkdf2_hash:
        raise SecureJSLoginError("test_hash != pbkdf2_hash")
    # log.debug("OK: test_hash == pbkdf2_hash")
    return True
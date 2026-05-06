def encode_password(password):
    """Performs URL encoding for passwords

    :param password: (str) password to encode
    :return: (str) encoded password
    """
    log = logging.getLogger(mod_logger + '.password_encoder')
    log.debug('Encoding password: {p}'.format(p=password))
    encoded_password = ''
    for c in password:
        encoded_password += encode_character(char=c)
    log.debug('Encoded password: {p}'.format(p=encoded_password))
    return encoded_password
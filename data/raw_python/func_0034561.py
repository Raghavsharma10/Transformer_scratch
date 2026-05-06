def _decode_signed_user(encoded_sig, encoded_data):
    """ Decodes the ``POST``ed signed data
    """
    decoded_sig = _decode(encoded_sig)
    decoded_data = loads(_decode(encoded_data))

    if decoded_sig != hmac.new(app.config['CANVAS_CLIENT_SECRET'], 
        encoded_data, sha256).digest():
        raise ValueError("sig doesn't match hash")

    return decoded_sig, decoded_data
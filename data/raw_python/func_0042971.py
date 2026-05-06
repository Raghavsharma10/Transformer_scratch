def decrypt(data, _key):
    """
    ACCEPT BYTES -> UTF8 -> JSON -> {"salt":s, "length":l, "data":d}
    """
    # Key and iv have not been generated or provided, bail out
    if _key is None:
        Log.error("Expecting a key")

    _input = get_module("mo_json").json2value(data.decode('utf8'), leaves=False, flexible=False)

    # Initialize encryption using key and iv
    key_expander_256 = key_expander.KeyExpander(256)
    expanded_key = key_expander_256.expand(_key)
    aes_cipher_256 = aes_cipher.AESCipher(expanded_key)
    aes_cbc_256 = cbc_mode.CBCMode(aes_cipher_256, 16)
    aes_cbc_256.set_iv(base642bytearray(_input.salt))

    raw = base642bytearray(_input.data)
    out_data = bytearray()
    for _, e in _groupby16(raw):
        out_data.extend(aes_cbc_256.decrypt_block(e))

    if _input.encoding:
        return binary_type(out_data[:_input.length:]).decode(_input.encoding)
    else:
        return binary_type(out_data[:_input.length:])
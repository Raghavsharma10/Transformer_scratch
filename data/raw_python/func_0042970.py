def encrypt(text, _key, salt=None):
    """
    RETURN {"salt":s, "length":l, "data":d} -> JSON -> UTF8
    """

    if is_text(text):
        encoding = 'utf8'
        data = bytearray(text.encode("utf8"))
    elif is_binary(text):
        encoding = None
        if PY2:
            data = bytearray(text)
        else:
            data = text

    if _key is None:
        Log.error("Expecting a key")
    if is_binary(_key):
        _key = bytearray(_key)
    if salt is None:
        salt = Random.bytes(16)

    # Initialize encryption using key and iv
    key_expander_256 = key_expander.KeyExpander(256)
    expanded_key = key_expander_256.expand(_key)
    aes_cipher_256 = aes_cipher.AESCipher(expanded_key)
    aes_cbc_256 = cbc_mode.CBCMode(aes_cipher_256, 16)
    aes_cbc_256.set_iv(salt)

    output = Data()
    output.type = "AES256"
    output.salt = bytes2base64(salt)
    output.length = len(data)
    output.encoding = encoding

    encrypted = bytearray()
    for _, d in _groupby16(data):
        encrypted.extend(aes_cbc_256.encrypt_block(d))
    output.data = bytes2base64(encrypted)
    json = get_module("mo_json").value2json(output, pretty=True).encode('utf8')

    if DEBUG:
        test = decrypt(json, _key)
        if test != text:
            Log.error("problem with encryption")

    return json
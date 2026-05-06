def _encrypt(data):
    """Equivalent to OpenSSL using 256 bit AES in CBC mode"""
    BS = AES.block_size

    def pad(s):
        n = BS - len(s) % BS
        char = chr(n).encode('utf8')
        return s + n * char

    password = settings.GECKOBOARD_PASSWORD
    salt = Random.new().read(BS - len('Salted__'))
    key, iv = _derive_key_and_iv(password, salt, 32, BS)
    cipher = AES.new(key, AES.MODE_CBC, iv)
    encrypted = b'Salted__' + salt + cipher.encrypt(pad(data))
    return base64.b64encode(encrypted)
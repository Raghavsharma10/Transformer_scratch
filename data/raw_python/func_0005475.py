def aes_encrypt(value, secret, block_size=AES.block_size):
    """ AES encrypt @value with @secret using the |CFB| mode of AES
        with a cryptographically secure initialization vector.

        -> (#bytes) AES encrypted @value

        ..
            from vital.security import aes_encrypt, aes_decrypt
            aes_encrypt("Hello, world",
                        "aLWEFlwgwlreWELFNWEFWLEgwklgbweLKWEBGW")
            # -> 'zYgVYMbeOuiHR50aMFinY9JsfyMQCvpzI+LNqNcmZhw='
            aes_decrypt(
                "zYgVYMbeOuiHR50aMFinY9JsfyMQCvpzI+LNqNcmZhw=",
                "aLWEFlwgwlreWELFNWEFWLEgwklgbweLKWEBGW")
            # -> 'Hello, world'
        ..f
    """
    iv = os.urandom(block_size * 2)
    cipher = AES.new(secret[:32], AES.MODE_CFB, iv[:block_size])
    return b'%s%s' % (iv, cipher.encrypt(value))
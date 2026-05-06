def aes_decrypt(value, secret, block_size=AES.block_size):
    """ AES decrypt @value with @secret using the |CFB| mode of AES
        with a cryptographically secure initialization vector.

        -> (#str) AES decrypted @value

        ..
            from vital.security import aes_encrypt, aes_decrypt
            aes_encrypt("Hello, world",
                        "aLWEFlwgwlreWELFNWEFWLEgwklgbweLKWEBGW")
            # -> 'zYgVYMbeOuiHR50aMFinY9JsfyMQCvpzI+LNqNcmZhw='
            aes_decrypt(
                "zYgVYMbeOuiHR50aMFinY9JsfyMQCvpzI+LNqNcmZhw=",
                "aLWEFlwgwlreWELFNWEFWLEgwklgbweLKWEBGW")
            # -> 'Hello, world'
        ..
    """
    if value is not None:
        cipher = AES.new(secret[:32], AES.MODE_CFB, value[:block_size])
        return cipher.decrypt(uniorbytes(value[block_size * 2:], bytes))
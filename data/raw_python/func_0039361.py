def encryptstring(text, password):
    """
    Encrypt a string according to a specific password.

    :type text: string
    :param text: The text to encrypt.

    :type pass: string
    :param pass: The password to encrypt the text with.
    """

    enc = []
    for i in enumerate(text):
        key_c = password[i[0] % len(password)]
        enc_c = chr((ord(i[1]) + ord(key_c)) % 256)
        enc.append(enc_c)
    return base64.urlsafe_b64encode("".join(enc).encode()).decode()
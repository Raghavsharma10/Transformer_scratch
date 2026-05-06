def decryptstring(enc, password):
    """
    Decrypt an encrypted string according to a specific password.

    :type enc: string
    :param enc: The encrypted text.

    :type pass: string
    :param pass: The password used to encrypt the text.
    """

    dec = []
    enc = base64.urlsafe_b64decode(enc).decode()
    for i in enumerate(enc):
        key_c = password[i[0] % len(password)]
        dec_c = chr((256 + ord(i[1]) - ord(key_c)) % 256)
        dec.append(dec_c)
    return "".join(dec)
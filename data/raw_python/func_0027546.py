def encry_decry_chunk(chunk, key, algo, bool_encry, assoc_data):
    """
    When bool_encry is True, encrypt a chunk of the file with the key and a randomly generated nonce. When it is False,
    the function extract the nonce from the cipherchunk (first 16 bytes), and decrypt the rest of the chunk.
    :param chunk: a chunk in bytes to encrypt or decrypt.
    :param key: a 32 bytes key in bytes.
    :param algo: a string of algorithm. Can be "srp" , "AES" or "twf"
    :param bool_encry: if bool_encry is True, chunk is encrypted. Else, it will be decrypted.
    :param assoc_data: bytes string of additional data for GCM Authentication.
    :return: if bool_encry is True, corresponding nonce + cipherchunk else, a decrypted chunk.
    """
    engine = botan.cipher(algo=algo, encrypt=bool_encry)
    engine.set_key(key=key)
    engine.set_assoc_data(assoc_data)
    if bool_encry is True:
        nonce = generate_nonce_timestamp()
        engine.start(nonce=nonce)
        return nonce + engine.finish(chunk)
    else:
        nonce = chunk[:__nonce_length__]
        encryptedchunk = chunk[__nonce_length__:__nonce_length__ + __gcmtag_length__ + __chunk_size__]
        engine.start(nonce=nonce)
        decryptedchunk = engine.finish(encryptedchunk)
        if decryptedchunk == b"":
            raise Exception("Integrity failure: Invalid passphrase or corrupted data")
        return decryptedchunk
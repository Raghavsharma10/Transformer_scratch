def encryptfile(filename, passphrase, algo='srp'):
    """
    Encrypt a file and write it with .cryptoshop extension.
    :param filename: a string with the path to the file to encrypt.
    :param passphrase: a string with the user passphrase.
    :param algo: a string with the algorithm. Can be srp, aes, twf. Default is srp.
    :return: a string with "successfully encrypted" or error.
    """
    try:
        if algo == "srp":
            header = b"Cryptoshop srp " + b_version
            crypto_algo = "Serpent/GCM"
        if algo == "aes":
            header = b"Cryptoshop aes " + b_version
            crypto_algo = "AES-256/GCM"
        if algo == "twf":
            header = b"Cryptoshop twf " + b_version
            crypto_algo = "Twofish/GCM"
        if algo != "srp" and algo != "aes" and algo != "twf":
            return "No valid algo. Use 'srp' 'aes' or 'twf'"
        outname = filename + ".cryptoshop"

        internal_key = botan.rng().get(internal_key_length)

        # Passphrase derivation...
        salt = botan.rng().get(__salt_size__)
        masterkey = calc_derivation(passphrase=passphrase, salt=salt)

        # Encrypt internal key...
        encrypted_key = encry_decry_cascade(data=internal_key, masterkey=masterkey,
                                            bool_encry=True,
                                            assoc_data=header)
        with open(filename, 'rb') as filestream:
            file_size = os.stat(filename).st_size
            if file_size == 0:
                raise Exception("Error: You can't encrypt empty file.")
            with open(str(outname), 'wb') as filestreamout:
                filestreamout.write(header)
                filestreamout.write(salt)
                filestreamout.write(encrypted_key)

                finished = False
                # the maximum of the progress bar is the total chunk to process. It's files_size // chunk_size
                bar = tqdm(range(file_size // __chunk_size__))
                while not finished:
                    chunk = filestream.read(__chunk_size__)
                    if len(chunk) == 0 or len(chunk) % __chunk_size__ != 0:
                        finished = True
                    # An encrypted-chunk output is nonce, gcmtag, and cipher-chunk concatenation.
                    encryptedchunk = encry_decry_chunk(chunk=chunk, key=internal_key, bool_encry=True,
                                                       algo=crypto_algo, assoc_data=header)
                    filestreamout.write(encryptedchunk)
                    bar.update(1)

            return "successfully encrypted"

    except IOError:
        exit("Error: file \"" + filename + "\" was not found.")
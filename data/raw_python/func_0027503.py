def decryptfile(filename, passphrase):
    """
    Decrypt a file and write corresponding decrypted file. We remove the .cryptoshop extension.
    :param filename: a string with the path to the file to decrypt.
    :param passphrase: a string with the user passphrase.
    :return: a string with "successfully decrypted" or error.
    """
    try:
        outname = os.path.splitext(filename)[0].split("_")[-1]  # create a string file name without extension.
        with open(filename, 'rb') as filestream:
            file_size = os.stat(filename).st_size
            if file_size == 0:
                raise Exception("Error: You can't decrypt empty file.")

            fileheader = filestream.read(header_length)

            if fileheader == b"Cryptoshop srp " + b_version:
                decrypt_algo = "Serpent/GCM"
            if fileheader == b"Cryptoshop aes " + b_version:
                decrypt_algo = "AES-256/GCM"
            if fileheader == b"Cryptoshop twf " + b_version:
                decrypt_algo = "Twofish/GCM"
            if fileheader != b"Cryptoshop srp " + b_version and fileheader != b"Cryptoshop aes " + b_version and fileheader != b"Cryptoshop twf " + b_version:
                raise Exception("Integrity failure: Bad header")

            salt = filestream.read(__salt_size__)
            encrypted_key = filestream.read(encrypted_key_length)

            # Derive the passphrase...
            masterkey = calc_derivation(passphrase=passphrase, salt=salt)

            # Decrypt internal key...
            try:
                internal_key = encry_decry_cascade(data=encrypted_key, masterkey=masterkey,
                                                   bool_encry=False, assoc_data=fileheader)
            except Exception as e:
                return e

            with open(str(outname), 'wb') as filestreamout:
                files_size = os.stat(filename).st_size
                # the maximum of the progress bar is the total chunk to process. It's files_size // chunk_size
                bar = tqdm(range(files_size // __chunk_size__))
                while True:
                    # Don't forget... an encrypted chunk is nonce, gcmtag, and cipher-chunk concatenation.
                    encryptedchunk = filestream.read(__nonce_length__ + __gcmtag_length__ + __chunk_size__)
                    if len(encryptedchunk) == 0:
                        break

                    # Chunk decryption.
                    try:
                        original = encry_decry_chunk(chunk=encryptedchunk, key=internal_key, algo=decrypt_algo,
                                                     bool_encry=False, assoc_data=fileheader)
                    except Exception as e:
                        return e
                    else:
                        filestreamout.write(original)
                        bar.update(1)

        return "successfully decrypted"

    except IOError:
        exit("Error: file \"" + filename + "\" was not found.")
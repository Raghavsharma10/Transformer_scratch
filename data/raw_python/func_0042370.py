def decrypt_file(file_path, recipient_key, *, base64=False):
    "Returns (filename, file_contents) if successful"
    crypto.assert_type_and_length('recipient_key', recipient_key, crypto.UserLock)
    with open(file_path, "rb") as I:
        contents = I.read()
        if base64:
            contents = crypto.b64decode(contents)
        crypted = crypto.MiniLockFile(contents)
    return crypted.decrypt(recipient_key)
def encrypt_file(file_path, sender, recipients):
    "Returns encrypted binary file content if successful"
    for recipient_key in recipients:
        crypto.assert_type_and_length('recipient_key', recipient_key, (str, crypto.UserLock))
    crypto.assert_type_and_length("sender_key", sender, crypto.UserLock)
    if (not os.path.exists(file_path)) or (not os.path.isfile(file_path)):
        raise OSError("Specified path does not point to a valid file: {}".format(file_path))
    _, filename = os.path.split(file_path)
    with open(file_path, "rb") as I:
        crypted = crypto.MiniLockFile.new(filename, I.read(), sender, recipients)
    return crypted.contents
def encrypt_folder(path, sender, recipients):
    """
    This helper function should zip the contents of a folder and encrypt it as
    a zip-file. Recipients are responsible for opening the zip-file.
    """
    for recipient_key in recipients:
        crypto.assert_type_and_length('recipient_key', recipient_key, (str, crypto.UserLock))
    crypto.assert_type_and_length("sender_key", sender, crypto.UserLock)
    if (not os.path.exists(path)) or (not os.path.isdir(path)):
        raise OSError("Specified path is not a valid directory: {}".format(path))
    buf = io.BytesIO()
    zipf = zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED)
    for root, folders, files in os.walk(path):
        for fn in files:
            fp = os.path.join(root, fn)
            zipf.write(fp)
    zipf.close()
    zip_contents = buf.getvalue()
    _, filename = os.path.split(path)
    filename += ".zip"
    crypted = crypto.MiniLockFile.new(filename, zip_contents, sender, recipients)
    return crypted.contents
def has_changed_since_last_deploy(file_path, bucket):
    """
    Checks if a file has changed since the last time it was deployed.

    :param file_path: Path to file which should be checked. Should be relative
                      from root of bucket.
    :param bucket_name: Name of S3 bucket to check against.
    :returns: True if the file has changed, else False.
    """

    msg = "Checking if {0} has changed since last deploy.".format(file_path)
    logger.debug(msg)
    with open(file_path) as f:
        data = f.read()
        file_md5 = hashlib.md5(data.encode('utf-8')).hexdigest()
        logger.debug("file_md5 is {0}".format(file_md5))

    key = bucket.get_key(file_path)

    # HACK: Boto's md5 property does not work when the file hasn't been
    # downloaded. The etag works but will break for multi-part uploaded files.
    # http://stackoverflow.com/questions/16872679/how-to-programmatically-
    #     get-the-md5-checksum-of-amazon-s3-file-using-boto/17607096#17607096
    # Also the double quotes around it must be stripped. Sketchy...boto's fault
    if key:
        key_md5 = key.etag.replace('"', '').strip()
        logger.debug("key_md5 is {0}".format(key_md5))
    else:
        logger.debug("File does not exist in bucket")
        return True

    if file_md5 == key_md5:
        logger.debug("File has not changed.")
        return False
    logger.debug("File has changed.")
    return True
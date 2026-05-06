def deploy_file(file_path, bucket):
    """ Uploads a file to an S3 bucket, as a public file. """

    # Paths look like:
    #  index.html
    #  css/bootstrap.min.css

    logger.info("Deploying {0}".format(file_path))

    # Upload the actual file to file_path
    k = Key(bucket)
    k.key = file_path
    try:
        k.set_contents_from_filename(file_path)
        k.set_acl('public-read')
    except socket.error:
        logger.warning("Caught socket.error while trying to upload {0}".format(
            file_path))
        msg = "Please file an issue with alotofeffort if you see this,"
        logger.warning(msg)
        logger.warning("providing as much info as you can.")
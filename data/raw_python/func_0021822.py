def s3_list(s3_bucket, s3_access_key_id, s3_secret_key, prefix=None):
    """ Lists the contents of the S3 bucket that end in .tbz and match
        the passed prefix, if any. """
    bucket = s3_connect(s3_bucket, s3_access_key_id, s3_secret_key)
    return sorted([key.name for key in bucket.list()
                   if key.name.endswith(".tbz")
                   and (prefix is None or key.name.startswith(prefix))])
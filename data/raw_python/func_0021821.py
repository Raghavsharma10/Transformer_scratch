def s3_connect(bucket_name, s3_access_key_id, s3_secret_key):
    """ Returns a Boto connection to the provided S3 bucket. """
    conn = connect_s3(s3_access_key_id, s3_secret_key)
    try:
        return conn.get_bucket(bucket_name)
    except S3ResponseError as e:
        if e.status == 403:
            raise Exception("Bad Amazon S3 credentials.")
        raise
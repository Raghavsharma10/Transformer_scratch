def s3_download(output_file_path, s3_bucket, s3_access_key_id, s3_secret_key,
                s3_file_key=None, prefix=None):
    """ Downloads the file matching the provided key, in the provided bucket,
        from Amazon S3.
        
        If s3_file_key is none, it downloads the last file
        from the provided bucket with the .tbz extension, filtering by
        prefix if it is provided. """
    bucket = s3_connect(s3_bucket, s3_access_key_id, s3_secret_key)
    if not s3_file_key:
        keys = s3_list(s3_bucket, s3_access_key_id, s3_secret_key, prefix)
        if not keys:
            raise Exception("Target S3 bucket is empty")
        s3_file_key = keys[-1]
    key = Key(bucket, s3_file_key)
    with open(output_file_path, "w+") as f:
        f.write(key.read())
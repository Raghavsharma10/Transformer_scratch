def get_file(path, s3_bucket=None):
    """Gets a file"""

    bucket_name = s3_bucket or oz.settings["s3_bucket"]

    if bucket_name:
        bucket = get_bucket(bucket_name)
        key = bucket.get_key(path)
        if not key:
            key = bucket.new_key(path)
        return S3File(key)
    else:
        return LocalFile(oz.settings["static_path"], path)
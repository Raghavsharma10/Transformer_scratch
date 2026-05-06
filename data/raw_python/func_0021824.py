def s3_upload(source_file_path, bucket_name, s3_access_key_id, s3_secret_key):
    """ Uploads the to Amazon S3 the contents of the provided file, keyed
        with the name of the file. """
    key = s3_key(bucket_name, s3_access_key_id, s3_secret_key)
    file_name = source_file_path.split("/")[-1]
    key.key = file_name
    if key.exists():
        raise Exception("s3 key %s already exists for current period."
                        % (file_name))
    key.set_contents_from_filename(source_file_path)
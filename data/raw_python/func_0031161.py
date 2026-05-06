def download_file(client, bucket, remote_path, local_path, overwrite=False):
    """Downloads a file from a bucket.
    
    TODO: docstring"""
    bucket = client.get_bucket(bucket)
    if (not overwrite) and os.path.isfile(local_path):
        raise OSError('File already exists!')
    with open(local_path, 'wb') as ofh:
        bucket.get_blob(remote_path).download_to_file(ofh)
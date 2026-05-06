def upload_file(client, bucket, local_path, remote_path, overwrite=False):
    """Uploads a file to a bucket.
    
    TODO: docstring"""
    bucket = client.get_bucket(bucket)
    blob = storage.Blob(remote_path, bucket)
    if (not overwrite) and blob.exists():
        raise Conflict('File/object already exists on the bucket!')
    blob.upload_from_filename(local_path)
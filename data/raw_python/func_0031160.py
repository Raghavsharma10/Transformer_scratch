def get_files(client, bucket, prefix=''):
    """Lists files/objects on a bucket.
    
    TODO: docstring"""
    bucket = client.get_bucket(bucket)
    files = list(bucket.list_blobs(prefix=prefix))    
    return files
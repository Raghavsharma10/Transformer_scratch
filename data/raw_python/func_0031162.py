def delete_file(client, bucket, remote_path):
    """Deletes a file from a bucket.
    
    TODO: docstring"""
    bucket = client.get_bucket(bucket)
    bucket.delete_blob(remote_path)
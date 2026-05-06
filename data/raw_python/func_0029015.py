def upload_file_to_s3(awsclient, bucket, key, filename):
    """Upload a file to AWS S3 bucket.

    :param awsclient:
    :param bucket:
    :param key:
    :param filename:
    :return:
    """
    client_s3 = awsclient.get_client('s3')
    transfer = S3Transfer(client_s3)
    # Upload /tmp/myfile to s3://bucket/key and print upload progress.
    transfer.upload_file(filename, bucket, key)
    response = client_s3.head_object(Bucket=bucket, Key=key)
    etag = response.get('ETag')
    version_id = response.get('VersionId', None)
    return etag, version_id
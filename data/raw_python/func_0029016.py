def remove_file_from_s3(awsclient, bucket, key):
    """Remove a file from an AWS S3 bucket.

    :param awsclient:
    :param bucket:
    :param key:
    :return:
    """
    client_s3 = awsclient.get_client('s3')
    response = client_s3.delete_object(Bucket=bucket, Key=key)
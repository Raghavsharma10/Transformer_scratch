def get_aws_s3_handle(config_map):
    """Convenience function for getting AWS S3 objects

    Added by cjshaw@mit.edu, Jan 9, 2015
    Added to aws_adapter build by birdland@mit.edu, Jan 25, 2015, and
    added support for Configuration
    May 25, 2017: Switch to boto3

    """
    url = 'https://' + config_map['s3_bucket'] + '.s3.amazonaws.com'
    if not AWS_CLIENT.is_aws_s3_client_set():
        client = boto3.client(
            's3',
            aws_access_key_id=config_map['put_public_key'],
            aws_secret_access_key=config_map['put_private_key']
        )
        AWS_CLIENT.set_aws_s3_client(client)
    else:
        client = AWS_CLIENT.s3
    return client, url
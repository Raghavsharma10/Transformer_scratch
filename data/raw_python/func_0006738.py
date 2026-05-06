def _connect_to_ec2(region, credentials):
    """
    :param region: The region of AWS to connect to.
    :param EC2Credentials credentials: The credentials to use to authenticate
        with EC2.

    :return: a connection object to AWS EC2
    """
    conn = boto.ec2.connect_to_region(
        region,
        aws_access_key_id=credentials.access_key_id,
        aws_secret_access_key=credentials.secret_access_key
    )
    if conn:
        return conn
    else:
        log_red('Failure to authenticate to EC2.')
        return False
def connect_to_ec2(region, access_key_id, secret_access_key):
    """ returns a connection object to AWS EC2  """
    conn = boto.ec2.connect_to_region(region,
                                      aws_access_key_id=access_key_id,
                                      aws_secret_access_key=secret_access_key)
    return conn
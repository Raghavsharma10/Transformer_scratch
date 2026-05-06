def save_ec2_state_locally(instance_id,
                           region,
                           username,
                           access_key_id,
                           secret_access_key):
    """ queries EC2 for details about a particular instance_id and
        stores those details locally
    """
    # retrieve the IP information from the instance
    data = get_ec2_info(instance_id,
                        region,
                        access_key_id,
                        secret_access_key,
                        username)
    return _save_state_locally(data)
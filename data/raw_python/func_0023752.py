def login_user_block(username, ssh_keys, create_password=True):
    """
    Helper function for creating Server.login_user blocks.

    (see: https://www.upcloud.com/api/8-servers/#create-server)
    """
    block = {
        'create_password': 'yes' if create_password is True else 'no',
        'ssh_keys': {
            'ssh_key': ssh_keys
        }
    }

    if username:
        block['username'] = username

    return block
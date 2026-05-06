def logout(lancet, service):
    """Forget saved passwords for the web services."""
    if service:
        services = [service]
    else:
        services = ['tracker', 'harvest']

    for service in services:
        url = lancet.config.get(service, 'url')
        key = 'lancet+{}'.format(url)
        username = lancet.config.get(service, 'username')
        with taskstatus('Logging out from {}', url) as ts:
            if keyring.get_password(key, username):
                keyring.delete_password(key, username)
                ts.ok('Logged out from {}', url)
            else:
                ts.ok('Already logged out from {}', url)
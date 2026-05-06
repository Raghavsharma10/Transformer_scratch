def _get_credentials(username=None, password=None, dbhost=None):
    """Obtain user credentials by arguments or asking the user"""

    # Database salt
    system_config = dbhost.objectmodels['systemconfig'].find_one({
        'active': True
    })

    try:
        salt = system_config.salt.encode('ascii')
    except (KeyError, AttributeError):
        log('No systemconfig or it is without a salt! '
            'Reinstall the system provisioning with'
            'hfos_manage.py install provisions -p system')
        sys.exit(3)

    if username is None:
        username = _ask("Please enter username: ")
    else:
        username = username

    if password is None:
        password = _ask_password()
    else:
        password = password

    try:
        password = password.encode('utf-8')
    except UnicodeDecodeError:
        password = password

    passhash = hashlib.sha512(password)
    passhash.update(salt)

    return username, passhash.hexdigest()
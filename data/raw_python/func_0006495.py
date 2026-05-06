def apt_add_key(keyid, keyserver='keyserver.ubuntu.com', log=False):
    """ trust a new PGP key related to a apt-repository """
    if log:
        log_green(
            'trusting keyid %s from %s' % (keyid, keyserver)
        )
    with settings(hide('warnings', 'running', 'stdout')):
        sudo('apt-key adv --keyserver %s --recv %s' % (keyserver, keyid))
    return True
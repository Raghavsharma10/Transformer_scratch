def load_site(srcdir, datadir, site_name=None):
    """
    Load configuration values for a site.

    Returns (port, address, site_url, passwords)
    """
    if site_name is None:
        site_name = 'primary'
    if not validate.valid_name(site_name):
        raise DatacatsError('{} is not a valid site name.'.format(site_name))

    cp = ConfigParser.SafeConfigParser()
    try:
        cp.read([srcdir + '/.datacats-environment'])
    except ConfigParser.Error:
        raise DatacatsError('Error reading environment information')

    site_section = 'site_' + site_name
    try:
        port = cp.getint(site_section, 'port')
    except (ConfigParser.NoOptionError, ConfigParser.NoSectionError):
        port = None
    try:
        address = cp.get(site_section, 'address')
    except (ConfigParser.NoOptionError, ConfigParser.NoSectionError):
        address = None
    try:
        site_url = cp.get(site_section, 'site_url')
    except (ConfigParser.NoOptionError, ConfigParser.NoSectionError):
        site_url = None

    passwords = {}
    cp = ConfigParser.SafeConfigParser()
    cp.read(datadir + '/sites/' + site_name + '/passwords.ini')
    try:
        pw_options = cp.options('passwords')
    except ConfigParser.NoSectionError:
        pw_options = []

    for n in pw_options:
        passwords[n.upper()] = cp.get('passwords', n)

    return port, address, site_url, passwords
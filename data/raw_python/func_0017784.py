def save_new_site(site_name, sitedir, srcdir, port, address, site_url,
        passwords):
    """
    Add a site's configuration to the source dir and site dir
    """
    cp = ConfigParser.SafeConfigParser()
    cp.read([srcdir + '/.datacats-environment'])

    section_name = 'site_' + site_name

    if not cp.has_section(section_name):
        cp.add_section(section_name)
    cp.set(section_name, 'port', str(port))
    if address:
        cp.set(section_name, 'address', address)

    if site_url:
        cp.set(section_name, 'site_url', site_url)

    with open(srcdir + '/.datacats-environment', 'w') as config:
        cp.write(config)

    # save passwords to datadir
    cp = ConfigParser.SafeConfigParser()

    cp.add_section('passwords')
    for n in sorted(passwords):
        cp.set('passwords', n.lower(), passwords[n])

    # Write to the sitedir so we maintain separate passwords.
    with open(sitedir + '/passwords.ini', 'w') as config:
        cp.write(config)
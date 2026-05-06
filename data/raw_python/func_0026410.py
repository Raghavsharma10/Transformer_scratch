def install_nginx(instance, dbhost, dbname, port, hostname=None):
    """Install nginx configuration"""

    _check_root()

    log("Installing nginx configuration")

    if hostname is None:
        try:
            configuration = _get_system_configuration(dbhost, dbname)
            hostname = configuration.hostname
        except Exception as e:
            log('Exception:', e, type(e), exc=True, lvl=error)
            log("""Could not determine public fully qualified hostname!
Check systemconfig (see db view and db modify commands) or specify
manually with --hostname host.domain.tld

Using 'localhost' for now""", lvl=warn)
            hostname = 'localhost'

    definitions = {
        'instance': instance,
        'server_public_name': hostname,
        'ssl_certificate': cert_file,
        'ssl_key': key_file,
        'host_url': 'http://127.0.0.1:%i/' % port
    }

    if distribution == 'DEBIAN':
        configuration_file = '/etc/nginx/sites-available/hfos.%s.conf' % instance
        configuration_link = '/etc/nginx/sites-enabled/hfos.%s.conf' % instance
    elif distribution == 'ARCH':
        configuration_file = '/etc/nginx/nginx.conf'
        configuration_link = None
    else:
        log('Unsure how to proceed, you may need to specify your '
            'distribution', lvl=error)
        return

    log('Writing nginx HFOS site definition')
    write_template_file(os.path.join('dev/templates', nginx_configuration),
                        configuration_file,
                        definitions)

    if configuration_link is not None:
        log('Enabling nginx HFOS site (symlink)')
        if not os.path.exists(configuration_link):
            os.symlink(configuration_file, configuration_link)

    log('Restarting nginx service')
    Popen([
        'systemctl',
        'restart',
        'nginx.service'
    ])

    log("Done: Install nginx configuration")
def load_conf(configfile, config=None):
    """Get authentication data from the AUTH_CONF file."""
    default_login = 'your-login-for-api-here'
    default_password = 'your-password-for-api-here'
    config = config or {}
    configfile = local.path(configfile)
    if not configfile.exists():
        configfile.dirname.mkdir()
    else:
        assert_secure_file(configfile)
    with secure_filestore(), cli.Config(configfile) as conf:
        config['url'] = conf.get('habitipy.url', 'https://habitica.com')
        config['login'] = conf.get('habitipy.login', default_login)
        config['password'] = conf.get('habitipy.password', default_password)
        if config['login'] == default_login or config['password'] == default_password:
            if cli.terminal.ask(
                    _("""Your creditentials are invalid. Do you want to enter them now?"""),
                    default=True):
                msg = _("""
                You can get your login information at
                https://habitica.com/#/options/settings/api
                Both your user id and API token should look like this:
                xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
                where 'x' is a number between 0-9 or a character a-f.
                """)
                print(dedent(msg))
                msg = _("""Please enter your login (user ID)""")
                config['login'] = cli.terminal.prompt(msg, validator=is_uuid)
                msg = _("""Please enter your password (API token)""")
                config['password'] = cli.terminal.prompt(msg, validator=is_uuid)
                conf.set('habitipy.login', config['login'])
                conf.set('habitipy.password', config['password'])
                print(dedent(_("""
                Your creditentials are securely stored in
                {configfile}
                You can edit that file later if you need.
                """)).format(configfile=configfile))
        config['show_numbers'] = conf.get('habitipy.show_numbers', 'y')
        config['show_numbers'] = config['show_numbers'] in YES_ANSWERS
        config['show_style'] = conf.get('habitipy.show_style', 'wide')
        if config['show_style'] not in CHECK_MARK_STYLES:
            config['show_style'] = 'wide'
    return config
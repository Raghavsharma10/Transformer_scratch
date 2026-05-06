def get_args_parser():
    """Return a parser for command line options."""
    parser = argparse.ArgumentParser(
        description='Marabunta: Migrating ants for Odoo')
    parser.add_argument('--migration-file', '-f',
                        action=EnvDefault,
                        envvar='MARABUNTA_MIGRATION_FILE',
                        required=True,
                        help='The yaml file containing the migration steps')
    parser.add_argument('--database', '-d',
                        action=EnvDefault,
                        envvar='MARABUNTA_DATABASE',
                        required=True,
                        help="Odoo's database")
    parser.add_argument('--db-user', '-u',
                        action=EnvDefault,
                        envvar='MARABUNTA_DB_USER',
                        required=True,
                        help="Odoo's database user")
    parser.add_argument('--db-password', '-w',
                        action=EnvDefault,
                        envvar='MARABUNTA_DB_PASSWORD',
                        required=True,
                        help="Odoo's database password")
    parser.add_argument('--db-port', '-p',
                        default=os.environ.get('MARABUNTA_DB_PORT', 5432),
                        help="Odoo's database port")
    parser.add_argument('--db-host', '-H',
                        default=os.environ.get('MARABUNTA_DB_HOST',
                                               'localhost'),
                        help="Odoo's database host")
    parser.add_argument('--mode',
                        action=EnvDefault,
                        envvar='MARABUNTA_MODE',
                        required=False,
                        help="Specify the mode in which we run the migration,"
                             "such as 'demo' or 'prod'. Additional operations "
                             "of this mode will be executed after the main "
                             "operations and the addons list of this mode "
                             "will be merged with the main addons list.")
    parser.add_argument('--allow-serie',
                        action=BoolEnvDefault,
                        required=False,
                        envvar='MARABUNTA_ALLOW_SERIE',
                        help='Allow to run more than 1 version upgrade at a '
                             'time.')
    parser.add_argument('--force-version',
                        required=False,
                        default=os.environ.get('MARABUNTA_FORCE_VERSION'),
                        help='Force upgrade of a version, even if it has '
                             'already been applied.')

    group = parser.add_argument_group(
        title='Web',
        description='Configuration related to the internal web server, '
                    'used to publish a maintenance page during the migration.',
    )
    group.add_argument('--web-host',
                       required=False,
                       default=os.environ.get('MARABUNTA_WEB_HOST', '0.0.0.0'),
                       help='Host for the web server')
    group.add_argument('--web-port',
                       required=False,
                       default=os.environ.get('MARABUNTA_WEB_PORT', 8069),
                       help='Port for the web server')
    group.add_argument('--web-custom-html',
                       required=False,
                       default=os.environ.get(
                           'MARABUNTA_WEB_CUSTOM_HTML'
                       ),
                       help='Path to a custom html file to publish')
    return parser
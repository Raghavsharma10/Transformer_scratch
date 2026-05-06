def main():
    """Run the bot."""
    args = parser.parse_args()
    initialize_logging(args)

    # Allow expansion of paths even if the shell doesn't do it
    config_path = os.path.abspath(os.path.expanduser(args.config))

    client = kitnirc.client.Client()
    controller = kitnirc.modular.Controller(client, config_path)

    # Make sure the configuration file is loaded so we can check for
    # connection information.
    controller.load_config()

    def config_or_none(section, value, integer=False, boolean=False):
        """Helper function to get values that might not be set."""
        if controller.config.has_option(section, value):
            if integer:
                return controller.config.getint(section, value)
            elif boolean:
                return controller.config.getboolean(section, value)
            return controller.config.get(section, value)
        return None

    # If host isn't specified on the command line, try from config file
    host = args.host or config_or_none("server", "host")
    if not host:
        parser.error(
            "IRC host must be specified if not in config file.")

    # If nick isn't specified on the command line, try from config file
    nick = args.nick or config_or_none("server", "nick")
    if not nick:
        parser.error(
            "Nick must be specified if not in config file.")

    # KitnIRC's default client will use port 6667 if nothing else is specified,
    # but since we want to potentially specify something else, we add that
    # fallback here ourselves.
    port = args.port or config_or_none("server", "port", integer=True) or 6667
    ssl = args.ssl or config_or_none("server", "ssl", boolean=True)
    password = args.password or config_or_none("server", "password")
    username = args.username or config_or_none("server", "username") or nick
    realname = args.realname or config_or_none("server", "realname") or username

    controller.start()
    client.connect(
        nick,
        host=host,
        port=port,
        username=username,
        realname=realname,
        password=password,
        ssl=ssl,
    )
    try:
        client.run()
    except KeyboardInterrupt:
        client.disconnect()
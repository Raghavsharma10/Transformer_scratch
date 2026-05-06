def _setup_parser(self):
        """Setup a configuration parser.

        Contains built in ``--system-config`` || ``-SC`` variable which is used
        to allow a user to set arguments in a configuration file which would
        then be processed from the "default" section of the provided file,
        assuming the file exists.

        :return: ``tuple``
        """
        # Set the prefix for environment variables
        ename = self.env_name.upper()
        env_name = '%s_CONFIG' % ename

        # Accept a config file
        conf_parser = argparse.ArgumentParser(add_help=False)
        conf_parser.add_argument(
            '--system-config',
            metavar='[FILE]',
            type=str,
            default=os.environ.get(env_name, None),
            help='Path to your Configuration file. This is an optional'
                 ' argument used to specify config. available as: env[%s]'
                 % env_name
        )
        known_args, remaining_argv = conf_parser.parse_known_args()
        conf_file = known_args.system_config
        if conf_file is not None:
            file_name = os.path.basename(conf_file)
            config = parse_ini.ConfigurationSetup(log_name=file_name)
            path_dir = os.path.dirname(conf_file)
            config.load_config(path=path_dir)
            config_args = config.config_args(section='default')
            known_args.__dict__.update(config_args)

        parser = argparse.ArgumentParser(
            parents=[conf_parser],
            usage=self.usage,
            description=self.description,
            epilog=self.epilog)

        # Setup for the positional Arguments
        if self.detail is not None:
            self.detail = '%s\n' % self.detail

        if 'subparsed_args' in self.arguments:
            subparser = parser.add_subparsers(
                title=self.title,
                metavar=self.detail
            )
        else:
            subparser = None

        return parser, subparser, remaining_argv
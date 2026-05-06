def create_parser(self, prog_name, subcommand):
        """
        Create and return the ``OptionParser`` which will be used to
        parse the arguments to this command.

        """
        parser = argparse.ArgumentParser(prog='%s %s' % (prog_name, subcommand), description=self.help)

        parser.add_argument('-v', '--verbosity', action='store', default=1, type=int, choices=range(4),
            help='Verbosity level; 0=minimal output, 1=normal output, 2=verbose output, 3=very verbose output'),
        parser.add_argument('--settings',
            help='The Python path to a settings module, e.g. "myproject.settings.main". '
            'If this isn\'t provided, the DJANGO_SETTINGS_MODULE environment variable will be used.'),
        parser.add_argument('--pythonpath',
            help='A directory to add to the Python path, e.g. "/home/djangoprojects/myproject".'),
        parser.add_argument('--traceback', action='store_true', help='Print traceback on exception'),

        subparsers = parser.add_subparsers(description='JavaScript command to execute')

        for subparser in self.subparsers:
            subparser(self, subparsers)

        return parser
def options(self, parser, env):
        """Register commandline options.
        """
        parser.add_option(
            "--epdb", action="store_true", dest="epdb_debugErrors",
            default=env.get('NOSE_EPDB', False),
            help="Drop into extended debugger on errors")
        parser.add_option(
            "--epdb-failures", action="store_true",
            dest="epdb_debugFailures",
            default=env.get('NOSE_EPDB_FAILURES', False),
            help="Drop into extended debugger on failures")
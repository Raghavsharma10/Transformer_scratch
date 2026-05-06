def run_from_cli(self, args):
        """Read arguments, run and print results.

        Args:
            args (dict): Arguments parsed by docopt.
        """
        if args['--dump-config']:
            self._config.print_config()
        else:
            stdout, stderr = self.lint(args['<path>'])
            self.print_results(stdout, stderr)
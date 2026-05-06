def print_variables(self):
        """Prints out magic variables available in config files
        alongside with their values and descriptions.
        May be useful for debugging.

        http://uwsgi-docs.readthedocs.io/en/latest/Configuration.html#magic-variables

        """
        print_out = partial(self.print_out, format_options='green')

        print_out('===== variables =====')

        for var, hint in self.vars.get_descriptions().items():
            print_out('    %' + var + ' = ' + var + ' = ' + hint.replace('%', '%%'))

        print_out('=====================')

        return self
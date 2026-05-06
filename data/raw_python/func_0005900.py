def print_stamp(self):
        """Prints out a stamp containing useful information,
        such as what and when has generated this configuration.

        """
        from . import VERSION

        print_out = partial(self.print_out, format_options='red')
        print_out('This configuration was automatically generated using')
        print_out('uwsgiconf v%s on %s' % ('.'.join(map(str, VERSION)), datetime.now().isoformat(' ')))

        return self
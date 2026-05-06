def get_subparser(self, name, dest="subcommand", **kwargs):
        """Get or create subparser."""
        if name not in self.children:
            # Create the subparser.
            subparsers = self._get_subparsers(dest)
            parser = subparsers.add_parser(name, **kwargs)
            self.children[name] = NamedParser(name, parser)
        return self.children[name]
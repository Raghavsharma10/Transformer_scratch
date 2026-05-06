def _get_subparsers(self, dest):
        """Get named subparsers."""
        if not self._subparsers:
            self._subparsers = self.parser.add_subparsers(dest=dest)
        elif self._subparsers.dest != dest:
            raise KeyError(
                "Subparser names mismatch. You can only create one subcommand.")
        return self._subparsers
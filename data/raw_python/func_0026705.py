def _setup_argparse(self):
        """Create `argparse` instance, and setup with appropriate parameters.
        """
        parser = argparse.ArgumentParser(
            prog='catalog', description='Parent Catalog class for astrocats.')

        subparsers = parser.add_subparsers(
            description='valid subcommands', dest='subcommand')

        # Data Import
        # -----------
        # Add the 'import' command, and related arguments
        self._add_parser_arguments_import(subparsers)

        # Git Subcommands
        # ---------------
        self._add_parser_arguments_git(subparsers)

        # Analyze Catalogs
        # ----------------
        # Add the 'analyze' command, and related arguments
        self._add_parser_arguments_analyze(subparsers)

        return parser
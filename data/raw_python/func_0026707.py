def _add_parser_arguments_git(self, subparsers):
        """Create a sub-parsers for git subcommands.
        """
        subparsers.add_parser(
            "git-clone",
            help="Clone all defined data repositories if they dont exist.")

        subparsers.add_parser(
            "git-push",
            help="Add all files to data repositories, commit, and push.")

        subparsers.add_parser(
            "git-pull",
            help="'Pull' all data repositories.")

        subparsers.add_parser(
            "git-reset-local",
            help="Hard reset all data repositories using local 'HEAD'.")

        subparsers.add_parser(
            "git-reset-origin",
            help="Hard reset all data repositories using 'origin/master'.")

        subparsers.add_parser(
            "git-status",
            help="Get the 'git status' of all data repositories.")

        return
def install(self):
        """Install the package I represent, without dependencies.

        Obey typical pip-install options passed in on the command line.

        """
        other_args = list(requirement_args(self._argv, want_other=True))
        archive_path = join(self._temp_path, self._downloaded_filename())
        # -U so it installs whether pip deems the requirement "satisfied" or
        # not. This is necessary for GitHub-sourced zips, which change without
        # their version numbers changing.
        run_pip(['install'] + other_args + ['--no-deps', '-U', archive_path])
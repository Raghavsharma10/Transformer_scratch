def init(self, basedir, config, sourcedir, targetdir, cwd='', commit=True):
        """
        Init project structure and configuration from given arguments

        Args:
            basedir (string): Project base directory used to prepend relative
                paths. If empty or equal to '.', it will be filled with current
                directory path.
            config (string): Settings file path.
            sourcedir (string): Source directory path.
            targetdir (string): Compiled files target directory path.

        Keyword Arguments:
            cwd (string): Current directory path to prepend base dir if empty.
            commit (bool): If ``False``, directory structure and settings file
                won't be created.

        Returns:
            dict: A dict containing expanded given paths.
        """
        if not basedir:
            basedir = '.'

        # Expand home directory if any
        abs_basedir, abs_config, abs_sourcedir, abs_targetdir = self.expand(
            basedir, config,
            sourcedir, targetdir,
            cwd
        )

        # Valid every paths are ok
        self.valid_paths(abs_config, abs_sourcedir, abs_targetdir)

        # Create required directory structure
        if commit:
            self.commit(sourcedir, targetdir, abs_config, abs_sourcedir,
                        abs_targetdir)

        return {
            'basedir': abs_basedir,
            'config': abs_config,
            'sourcedir': abs_sourcedir,
            'targetdir': abs_targetdir,
        }
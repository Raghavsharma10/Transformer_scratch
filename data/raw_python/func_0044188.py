def expand(self, basedir, config, sourcedir, targetdir, cwd):
        """
        Validate that given paths are not the same.

        Args:
            basedir (string): Project base directory used to prepend relative
                paths. If empty or equal to '.', it will be filled with current
                directory path.
            config (string): Settings file path.
            sourcedir (string): Source directory path.
            targetdir (string): Compiled files target directory path.
            cwd (string): Current directory path to prepend base dir if empty.

        Returns:
            tuple: Expanded arguments in the same order
        """

        # Expand home directory if any
        expanded_basedir = os.path.expanduser(basedir)
        expanded_config = os.path.expanduser(config)
        expanded_sourcedir = os.path.expanduser(sourcedir)
        expanded_targetdir = os.path.expanduser(targetdir)

        # If not absolute, base dir is prepended with current directory
        if not os.path.isabs(expanded_basedir):
            expanded_basedir = os.path.join(cwd, expanded_basedir)
        # Prepend paths with base dir if they are not allready absolute
        if not os.path.isabs(expanded_config):
            expanded_config = os.path.join(expanded_basedir,
                                           expanded_config)
        if not os.path.isabs(expanded_sourcedir):
            expanded_sourcedir = os.path.join(expanded_basedir,
                                              expanded_sourcedir)
        if not os.path.isabs(expanded_targetdir):
            expanded_targetdir = os.path.join(expanded_basedir,
                                              expanded_targetdir)

        # Normalize paths
        expanded_basedir = os.path.normpath(expanded_basedir)
        expanded_config = os.path.normpath(expanded_config)
        expanded_sourcedir = os.path.normpath(expanded_sourcedir)
        expanded_targetdir = os.path.normpath(expanded_targetdir)

        return (expanded_basedir, expanded_config, expanded_sourcedir,
                expanded_targetdir)
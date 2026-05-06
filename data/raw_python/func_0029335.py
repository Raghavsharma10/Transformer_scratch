def discover(self, start, top_level_directory=None, pattern='test*.py'):
        """Do test case discovery.

        This is the top-level entry-point for test discovery.

        If the ``start`` argument is a drectory, then ``haas`` will
        discover all tests in the package contained in that directory.

        If the ``start`` argument is not a directory, it is assumed to
        be a package or module name and tests in the package or module
        are loaded.

        FIXME: This needs a better description.

        Parameters
        ----------
        start : str
            The directory, package, module, class or test to load.
        top_level_directory : str
            The path to the top-level directoy of the project.  This is
            the parent directory of the project'stop-level Python
            package.
        pattern : str
            The glob pattern to match the filenames of modules to search
            for tests.

        """
        logger.debug('Starting test discovery')
        if os.path.isdir(start):
            start_directory = start
            return self.discover_by_directory(
                start_directory, top_level_directory=top_level_directory,
                pattern=pattern)
        elif os.path.isfile(start):
            start_filepath = start
            return self.discover_by_file(
                start_filepath, top_level_directory=top_level_directory)
        else:
            package_or_module = start
            return self.discover_by_module(
                package_or_module, top_level_directory=top_level_directory,
                pattern=pattern)
def discover_by_directory(self, start_directory, top_level_directory=None,
                              pattern='test*.py'):
        """Run test discovery in a directory.

        Parameters
        ----------
        start_directory : str
            The package directory in which to start test discovery.
        top_level_directory : str
            The path to the top-level directoy of the project.  This is
            the parent directory of the project'stop-level Python
            package.
        pattern : str
            The glob pattern to match the filenames of modules to search
            for tests.

        """
        start_directory = os.path.abspath(start_directory)
        if top_level_directory is None:
            top_level_directory = find_top_level_directory(
                start_directory)
        logger.debug('Discovering tests in directory: start_directory=%r, '
                     'top_level_directory=%r, pattern=%r', start_directory,
                     top_level_directory, pattern)

        assert_start_importable(top_level_directory, start_directory)

        if top_level_directory not in sys.path:
            sys.path.insert(0, top_level_directory)
        tests = self._discover_tests(
            start_directory, top_level_directory, pattern)
        return self._loader.create_suite(list(tests))
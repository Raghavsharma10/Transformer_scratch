def discover_by_file(self, start_filepath, top_level_directory=None):
        """Run test discovery on a single file.

        Parameters
        ----------
        start_filepath : str
            The module file in which to start test discovery.
        top_level_directory : str
            The path to the top-level directoy of the project.  This is
            the parent directory of the project'stop-level Python
            package.

        """
        start_filepath = os.path.abspath(start_filepath)
        start_directory = os.path.dirname(start_filepath)
        if top_level_directory is None:
            top_level_directory = find_top_level_directory(
                start_directory)
        logger.debug('Discovering tests in file: start_filepath=%r, '
                     'top_level_directory=', start_filepath,
                     top_level_directory)

        assert_start_importable(top_level_directory, start_directory)

        if top_level_directory not in sys.path:
            sys.path.insert(0, top_level_directory)
        tests = self._load_from_file(
            start_filepath, top_level_directory)
        return self._loader.create_suite(list(tests))
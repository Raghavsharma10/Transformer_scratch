def load(self, files):
        """Load any files found into a suite.

        Any directories are walked and their files are added as TAP files.

        :returns: A ``unittest.TestSuite`` instance
        """
        suite = unittest.TestSuite()
        for filepath in files:
            if os.path.isdir(filepath):
                self._find_tests_in_directory(filepath, suite)
            else:
                suite.addTest(self.load_suite_from_file(filepath))
        return suite
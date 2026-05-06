def load_suite_from_file(self, filename):
        """Load a test suite with test lines from the provided TAP file.

        :returns: A ``unittest.TestSuite`` instance
        """
        suite = unittest.TestSuite()
        rules = Rules(filename, suite)

        if not os.path.exists(filename):
            rules.handle_file_does_not_exist()
            return suite

        line_generator = self._parser.parse_file(filename)
        return self._load_lines(filename, line_generator, suite, rules)
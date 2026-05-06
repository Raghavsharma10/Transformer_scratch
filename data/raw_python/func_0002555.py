def load_suite_from_stdin(self):
        """Load a test suite with test lines from the TAP stream on STDIN.

        :returns: A ``unittest.TestSuite`` instance
        """
        suite = unittest.TestSuite()
        rules = Rules("stream", suite)
        line_generator = self._parser.parse_stdin()
        return self._load_lines("stream", line_generator, suite, rules)
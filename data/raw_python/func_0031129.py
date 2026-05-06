def expect_regex(self, pattern, timeout=3, regex_options=0):
        """Wait for a match to the regex in *pattern* to appear on the stream.

        Waits for input matching the regex *pattern* for up to *timeout*
        seconds. If a match is found, a :class:`RegexMatch` result is returned.
        If no match is found within *timeout* seconds, raise an
        :class:`ExpectTimeout` exception.

        :param pattern: The pattern to search for, as a single compiled regex
            or a string that will be processed as a regex.
        :param float timeout: Timeout in seconds.
        :param regex_options: Options passed to the regex engine.
        :return: :class:`RegexMatch` if matched, None if no match was found.
        """
        return self.expect(RegexSearcher(pattern, regex_options), timeout)
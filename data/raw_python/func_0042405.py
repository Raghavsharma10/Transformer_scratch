def editline_with_regex(self, regex_tgtline, to_replace):
        """find the first matched line, then replace

        Args:
            regex_tgtline (str): regular expression used to match the target line
            to_replace    (str): line you wanna use to replace

        """
        for idx, line in enumerate(self._swp_lines):
            mobj = re.match(regex_tgtline, line)

            if mobj:
                self._swp_lines[idx] = to_replace

                return
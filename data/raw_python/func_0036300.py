def is_valid_line(self, line):
        """
        Validates a given line against the associated "section" (e.g. 'global'
        or 'frontend', etc.) of a stanza.

        If a line represents a directive that shouldn't be within the stanza
        it is rejected.  See the `directives.json` file for a condensed look
        at valid directives based on section.
        """
        adjusted_line = line.strip().lower()

        return any([
            adjusted_line.startswith(directive)
            for directive in directives_by_section[self.section_name]
        ])
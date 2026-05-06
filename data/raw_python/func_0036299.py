def add_line(self, line):
        """
        Adds a given line string to the list of lines, validating the line
        first.
        """
        if not self.is_valid_line(line):
            logger.warn(
                "Invalid line for %s section: '%s'",
                self.section_name, line
            )
            return

        self.lines.append(line)
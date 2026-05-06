def parse_data(self, logfile):
        """Parse data from data stream and replace object lines.

        :param logfile: [required] Log file data stream.
        :type logfile: str
        """

        for line in logfile:
            stripped_line = line.strip()
            parsed_line = Line(stripped_line)

            if parsed_line.valid:
                self._valid_lines.append(parsed_line)
            else:
                self._invalid_lines.append(stripped_line)
        self.total_lines = len(self._valid_lines) + len(self._invalid_lines)
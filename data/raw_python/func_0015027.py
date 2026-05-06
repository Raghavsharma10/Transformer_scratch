def cmd_status_codes_counter(self):
        """Generate statistics about HTTP status codes. 404, 500 and so on.
        """
        status_codes = defaultdict(int)
        for line in self._valid_lines:
            status_codes[line.status_code] += 1
        return status_codes
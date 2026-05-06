def cmd_ip_counter(self):
        """Reports a breakdown of how many requests have been made per IP.

        .. note::
          To enable this command requests need to provide a header with the
          forwarded IP (usually X-Forwarded-For) and be it the only header
          being captured.
        """
        ip_counter = defaultdict(int)
        for line in self._valid_lines:
            ip = line.get_ip()
            if ip is not None:
                ip_counter[ip] += 1
        return ip_counter
def cmd_http_methods(self):
        """Reports a breakdown of how many requests have been made per HTTP
        method (GET, POST...).
        """
        methods = defaultdict(int)
        for line in self._valid_lines:
            methods[line.http_request_method] += 1
        return methods
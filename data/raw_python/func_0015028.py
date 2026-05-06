def cmd_request_path_counter(self):
        """Generate statistics about HTTP requests' path."""
        paths = defaultdict(int)
        for line in self._valid_lines:
            paths[line.http_request_path] += 1
        return paths
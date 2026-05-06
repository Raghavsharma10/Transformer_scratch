def cmd_server_load(self):
        """Generate statistics regarding how many requests were processed by
        each downstream server.
        """
        servers = defaultdict(int)
        for line in self._valid_lines:
            servers[line.server_name] += 1
        return servers
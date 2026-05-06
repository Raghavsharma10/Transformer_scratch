def _add_index_server(self):
        """Adds index-server to 'distutil's 'index-servers' param."""
        index_servers = '\n\t'.join(self.servers.keys())
        self.conf.set('distutils', 'index-servers', index_servers)
def _get_index_servers(self):
        """Gets index-servers current configured in pypirc."""
        idx_srvs = []
        if 'index-servers' in self.conf.options('distutils'):
            idx = self.conf.get('distutils', 'index-servers')
            idx_srvs = [srv.strip() for srv in idx.split('\n') if srv.strip()]
        return idx_srvs
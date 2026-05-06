def save(self):
        """Saves pypirc file with new configuration information."""
        for server, conf in self.servers.iteritems():
            self._add_index_server()
            for conf_k, conf_v in conf.iteritems():
                if not self.conf.has_section(server):
                    self.conf.add_section(server)
                self.conf.set(server, conf_k, conf_v)

        with open(self.rc_file, 'wb') as configfile:
            self.conf.write(configfile)
        self.conf.read(self.rc_file)
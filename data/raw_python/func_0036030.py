def _nodemap_changed(self, data, stat):
        """Called when the nodemap changes."""

        if not stat:
            raise EnvironmentNotFoundException(self.nodemap_path)

        try:
            conf_path = self._deserialize_nodemap(data)[self.hostname]
        except KeyError:
            conf_path = '/services/%s/conf' % self.service

        self.config_watcher = DataWatch(
            self.zk, conf_path,
            self._config_changed
        )
def sync_file(self, clusters):
        """
        Generates new HAProxy config file content and writes it to the
        file at `haproxy_config_path`.

        If a restart is not necessary the nodes configured in HAProxy will
        be synced on the fly.  If a restart *is* necessary, one will be
        triggered.
        """
        logger.info("Updating HAProxy config file.")
        if not self.restart_required:
            self.sync_nodes(clusters)

        version = self.control.get_version()

        with open(self.haproxy_config_path, "w") as f:
            f.write(self.config_file.generate(clusters, version=version))

        if self.restart_required:
            with self.restart_lock:
                self.restart()
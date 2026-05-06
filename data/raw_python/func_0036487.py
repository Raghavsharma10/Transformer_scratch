def apply_config(self, config):
        """
        Takes the given config dictionary and sets the hosts and base_path
        attributes.

        If the kazoo client connection is established, its hosts list is
        updated to the newly configured value.
        """
        self.hosts = config["hosts"]
        old_base_path = self.base_path
        self.base_path = config["path"]
        if not self.connected.is_set():
            return

        logger.debug("Setting ZK hosts to %s", self.hosts)
        self.client.set_hosts(",".join(self.hosts))

        if old_base_path and old_base_path != self.base_path:
            logger.critical(
                "ZNode base path changed!" +
                " Lighthouse will need to be restarted" +
                " to watch the right znodes"
            )
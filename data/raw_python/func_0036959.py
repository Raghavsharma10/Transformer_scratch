def on_cluster_update(self, name, new_config):
        """
        Callback hook for when a cluster is updated.

        Or main concern when a cluster is updated is whether or not the
        associated discovery method changed.  If it did, we make sure that
        the old discovery method stops watching for the cluster's changes (if
        the old method is around) and that the new method *starts* watching
        for the cluster's changes (if the new method is actually around).

        Regardless of how the discovery method shuffling plays out the
        `sync_balancer_files` method is called.
        """
        cluster = self.configurables[Cluster][name]

        old_discovery = cluster.discovery
        new_discovery = new_config["discovery"]
        if old_discovery == new_discovery:
            self.sync_balancer_files()
            return

        logger.info(
            "Switching '%s' cluster discovery from '%s' to '%s'",
            name, old_discovery, new_discovery
        )

        if old_discovery in self.configurables[Discovery]:
            self.configurables[Discovery][old_discovery].stop_watching(
                cluster
            )
            self.kill_thread(cluster.name)
        if new_discovery not in self.configurables[Discovery]:
            logger.warn(
                "New discovery '%s' for cluster '%s' is unknown/unavailable.",
                new_discovery, name
            )
            self.sync_balancer_files()
            return

        discovery = self.configurables[Discovery][new_discovery]
        self.launch_thread(
            cluster.name,
            discovery.start_watching, cluster, self.sync_balancer_files
        )
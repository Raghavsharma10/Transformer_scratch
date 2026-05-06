def on_discovery_add(self, discovery):
        """
        When a discovery is added we call `connect()` on it and launch a thread
        for each cluster where the discovery watches for changes to the
        cluster's nodes.
        """
        discovery.connect()

        for cluster in self.configurables[Cluster].values():
            if cluster.discovery != discovery.name:
                continue

            self.launch_thread(
                cluster.name, discovery.start_watching,
                cluster, self.sync_balancer_files
            )

        self.sync_balancer_files()
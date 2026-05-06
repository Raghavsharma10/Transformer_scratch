def on_cluster_add(self, cluster):
        """
        Once a cluster is added we tell its associated discovery method to
        start watching for changes to the cluster's child nodes (if the
        discovery method is configured and available).
        """
        if cluster.discovery not in self.configurables[Discovery]:
            return

        discovery = self.configurables[Discovery][cluster.discovery]

        self.launch_thread(
            cluster.name, discovery.start_watching,
            cluster, self.sync_balancer_files
        )
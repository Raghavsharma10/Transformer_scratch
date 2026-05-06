def on_cluster_remove(self, name):
        """
        Stops the cluster's associated discovery method from watching for
        changes to the cluster's nodes.
        """
        discovery_name = self.configurables[Cluster][name].discovery
        if discovery_name in self.configurables[Discovery]:
            self.configurables[Discovery][discovery_name].stop_watching(
                self.configurables[Cluster][name]
            )
            self.kill_thread(name)

        self.sync_balancer_files()
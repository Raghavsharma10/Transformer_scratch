def stop_watching(self, cluster):
        """
        Causes the thread that launched the watch of the cluster path
        to end by setting the proper stop event found in `self.stop_events`.
        """
        znode_path = "/".join([self.base_path, cluster.name])
        if znode_path in self.stop_events:
            self.stop_events[znode_path].set()
def on_discovery_remove(self, name):
        """
        When a Discovery is removed we must make sure to call its `stop()`
        method to close any connections or do any clean up.
        """
        self.configurables[Discovery][name].stop()

        self.sync_balancer_files()
def start_watching(self, cluster, callback):
        """
        Initiates the "watching" of a cluster's associated znode.

        This is done via kazoo's ChildrenWatch object.  When a cluster's
        znode's child nodes are updated, a callback is fired and we update
        the cluster's `nodes` attribute based on the existing child znodes
        and fire a passed-in callback with no arguments once done.

        If the cluster's znode does not exist we wait for `NO_NODE_INTERVAL`
        seconds before trying again as long as no ChildrenWatch exists for
        the given cluster yet and we are not in the process of shutting down.
        """
        logger.debug("starting to watch cluster %s", cluster.name)
        wait_on_any(self.connected, self.shutdown)
        logger.debug("done waiting on (connected, shutdown)")
        znode_path = "/".join([self.base_path, cluster.name])

        self.stop_events[znode_path] = threading.Event()

        def should_stop():
            return (
                znode_path not in self.stop_events or
                self.stop_events[znode_path].is_set() or
                self.shutdown.is_set()
            )

        while not should_stop():
            try:
                if self.client.exists(znode_path):
                    break
            except exceptions.ConnectionClosedError:
                break

            wait_on_any(
                self.stop_events[znode_path], self.shutdown,
                timeout=NO_NODE_INTERVAL
            )

        logger.debug("setting up ChildrenWatch for %s", znode_path)

        @self.client.ChildrenWatch(znode_path)
        def watch(children):
            if should_stop():
                return False

            logger.debug("znode children changed! (%s)", znode_path)

            new_nodes = []
            for child in children:
                child_path = "/".join([znode_path, child])
                try:
                    new_nodes.append(
                        Node.deserialize(self.client.get(child_path)[0])
                    )
                except ValueError:
                    logger.exception("Invalid node at path '%s'", child)
                    continue

            cluster.nodes = new_nodes

            callback()
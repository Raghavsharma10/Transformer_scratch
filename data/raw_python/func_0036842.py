def sync_nodes(self, clusters):
        """
        Syncs the enabled/disabled status of nodes existing in HAProxy based
        on the given clusters.

        This is used to inform HAProxy of up/down nodes without necessarily
        doing a restart of the process.
        """
        logger.info("Syncing HAProxy backends.")

        current_nodes, enabled_nodes = self.get_current_nodes(clusters)

        for cluster_name, nodes in six.iteritems(current_nodes):
            for node in nodes:
                if node["svname"] in enabled_nodes[cluster_name]:
                    command = self.control.enable_node
                else:
                    command = self.control.disable_node

                try:
                    response = command(cluster_name, node["svname"])
                except Exception:
                    logger.exception("Error when enabling/disabling node")
                    self.restart_required = True
                else:
                    if response:
                        logger.error(
                            "Socket command for %s node %s failed: %s",
                            cluster_name, node["svname"], response
                        )
                        self.restart_required = True
                        return

        logger.info("HAProxy nodes/servers synced.")
def get_current_nodes(self, clusters):
        """
        Returns two dictionaries, the current nodes and the enabled nodes.

        The current_nodes dictionary is keyed off of the cluster name and
        values are a list of nodes known to HAProxy.

        The enabled_nodes dictionary is also keyed off of the cluster name
        and values are list of *enabled* nodes, i.e. the same values as
        current_nodes but limited to servers currently taking traffic.
        """
        current_nodes = self.control.get_active_nodes()
        enabled_nodes = collections.defaultdict(list)

        for cluster in clusters:
            if not cluster.nodes:
                continue

            if cluster.name not in current_nodes:
                logger.debug(
                    "New cluster '%s' added, restart required.",
                    cluster.name
                )
                self.restart_required = True

            for node in cluster.nodes:
                if node.name not in [
                        current_node["svname"]
                        for current_node in current_nodes.get(cluster.name, [])
                ]:
                    logger.debug(
                        "New node added to cluster '%s', restart required.",
                        cluster.name
                    )
                    self.restart_required = True

                enabled_nodes[cluster.name].append(node.name)

        return current_nodes, enabled_nodes
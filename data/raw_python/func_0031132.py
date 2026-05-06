def get_app_nodes(self, app_label):
        """
        Get all nodes for given app
        :param str app_label: app label
        :rtype: list
        """
        return [node for node in self.graph.nodes if node[0] == app_label]
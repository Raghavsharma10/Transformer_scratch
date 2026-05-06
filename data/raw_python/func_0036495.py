def path_of(self, service, node):
        """
        Helper method for determining the Zookeeper path for a given cluster
        member node.
        """
        return "/".join([self.base_path, service.name, node.name])
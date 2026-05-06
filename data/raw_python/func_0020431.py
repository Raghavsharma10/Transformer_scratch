def get_platform_node_selector(self, platform):
        """
        search the configuration for entries of the form node_selector.platform
        :param platform: str, platform to search for, can be null
        :return dict
        """
        nodeselector = {}
        if platform:
            nodeselector_str = self._get_value("node_selector." + platform, self.conf_section,
                                               "node_selector." + platform)
            nodeselector = self.generate_nodeselector_dict(nodeselector_str)

        return nodeselector
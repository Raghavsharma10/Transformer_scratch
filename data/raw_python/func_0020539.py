def get_config_map(self, name):
        """
        Get a ConfigMap object from the server

        Raises exception on error

        :param name: str, name of configMap to get from the server
        :returns: ConfigMapResponse containing the ConfigMap with the requested name
        """
        response = self.os.get_config_map(name)
        config_map_response = ConfigMapResponse(response.json())
        return config_map_response
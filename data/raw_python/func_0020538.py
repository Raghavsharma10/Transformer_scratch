def create_config_map(self, name, data):
        """
        Create an ConfigMap object on the server

        Raises exception on error

        :param name: str, name of configMap
        :param data: dict, dictionary of data to be stored
        :returns: ConfigMapResponse containing the ConfigMap with name and data
        """
        config_data_file = os.path.join(self.os_conf.get_build_json_store(), 'config_map.json')
        with open(config_data_file) as f:
            config_data = json.load(f)
        config_data['metadata']['name'] = name
        data_dict = {}
        for key, value in data.items():
            data_dict[key] = json.dumps(value)
        config_data['data'] = data_dict

        response = self.os.create_config_map(config_data)
        config_map_response = ConfigMapResponse(response.json())
        return config_map_response
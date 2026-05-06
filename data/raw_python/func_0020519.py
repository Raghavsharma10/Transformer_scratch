def get_data(self):
        """
        Find the data stored in the config_map

        :return: dict, the json of the data data that was passed into the ConfigMap on creation
        """
        data = graceful_chain_get(self.json, "data")
        if data is None:
            return {}

        data_dict = {}
        for key in data:
            if self.is_yaml(key):
                data_dict[key] = yaml.load(data[key])
            else:
                data_dict[key] = json.loads(data[key])

        return data_dict
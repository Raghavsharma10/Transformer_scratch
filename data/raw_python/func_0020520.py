def get_data_by_key(self, name):
        """
        Find the object stored by a JSON string at key 'name'

        :return: str or dict, the json of the str or dict stored in the ConfigMap at that location
        """
        data = graceful_chain_get(self.json, "data")

        if data is None or name not in data:
            return {}

        if self.is_yaml(name):
            return yaml.load(data[name]) or {}
        return json.loads(data[name])
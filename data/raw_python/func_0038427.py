def to_json(self, pretty=True):
        """
        to_json will call to_dict then dumps into json format
        """
        data_dict = self.to_dict()
        if pretty:
            return json.dumps(
                data_dict, sort_keys=True, indent=2)
        return json.dumps(data_dict, sort_keys=True)
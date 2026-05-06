def save_swagger_spec(self, filepath=None):
        """
        Saves a copy of the origin_spec to a local file in JSON format
        """
        if filepath is True or filepath is None:
            filepath = self.file_spec.format(server=self.server)

        json.dump(self.origin_spec, open(filepath, 'w+'), indent=3)
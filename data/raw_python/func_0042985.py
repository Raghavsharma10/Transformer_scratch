def load_swagger_spec(self, filepath=None):
        """
        Loads the origin_spec from a local JSON file.  If `filepath`
        is not provided, then the class `file_spec` format will be used
        to create the file-path value.
        """
        if filepath is True or filepath is None:
            filepath = self.file_spec.format(server=self.server)

        return json.load(open(filepath))
def register_schema(self, directory, path):
        """Register a json-schema.

        :param directory: root directory path.
        :param path: schema path, relative to the root directory.
        """
        self.schemas[path] = os.path.abspath(directory)
def get_schema_dir(self, path):
        """Retrieve the directory containing the given schema.

        :param path: Schema path, relative to the directory where it was
            registered.
        :raises invenio_jsonschemas.errors.JSONSchemaNotFound: If no schema
            was found in the specified path.
        :returns: The schema directory.
        """
        if path not in self.schemas:
            raise JSONSchemaNotFound(path)
        return self.schemas[path]
def get_schema_path(self, path):
        """Compute the schema's absolute path from a schema relative path.

        :param path: relative path of the schema.
        :raises invenio_jsonschemas.errors.JSONSchemaNotFound: If no schema
            was found in the specified path.
        :returns: The absolute path.
        """
        if path not in self.schemas:
            raise JSONSchemaNotFound(path)
        return os.path.join(self.schemas[path], path)
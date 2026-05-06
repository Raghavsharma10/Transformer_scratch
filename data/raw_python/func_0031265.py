def get_schema(self, path, with_refs=False, resolved=False):
        """Retrieve a schema.

        :param path: schema's relative path.
        :param with_refs: replace $refs in the schema.
        :param resolved: resolve schema using the resolver
            :py:const:`invenio_jsonschemas.config.JSONSCHEMAS_RESOLVER_CLS`
        :raises invenio_jsonschemas.errors.JSONSchemaNotFound: If no schema
            was found in the specified path.
        :returns: The schema in a dictionary form.
        """
        if path not in self.schemas:
            raise JSONSchemaNotFound(path)
        with open(os.path.join(self.schemas[path], path)) as file_:
            schema = json.load(file_)
            if with_refs:
                schema = JsonRef.replace_refs(
                    schema,
                    base_uri=request.base_url,
                    loader=self.loader_cls() if self.loader_cls else None,
                )
            if resolved:
                schema = self.resolver_cls(schema)
            return schema
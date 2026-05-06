def get_schema_object(self, fully_qualified_name: str) -> 'BaseSchema':
        """
        Used to generate a schema object from the given fully_qualified_name.
        :param fully_qualified_name: The fully qualified name of the object needed.
        :return: An initialized schema object
        """

        if fully_qualified_name not in self._schema_cache:
            spec = self.get_schema_spec(fully_qualified_name)

            if spec:
                try:
                    self._schema_cache[fully_qualified_name] = TypeLoader.load_schema(
                        spec.get(ATTRIBUTE_TYPE, None))(fully_qualified_name, self)
                except TypeLoaderError as err:
                    self.add_errors(
                        InvalidTypeError(fully_qualified_name, spec, ATTRIBUTE_TYPE,
                                         InvalidTypeError.Reason.TYPE_NOT_LOADED,
                                         err.type_class_name))

        return self._schema_cache.get(fully_qualified_name, None)
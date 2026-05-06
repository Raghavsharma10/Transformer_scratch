def get_schema_spec(self, fully_qualified_name: str) -> Dict[str, Any]:
        """
        Used to retrieve the specifications of the schema from the given
        fully_qualified_name of schema.
        :param fully_qualified_name: The fully qualified name of the schema needed.
        :return: Schema dictionary.
        """

        if fully_qualified_name not in self._spec_cache:
            self.add_errors(SpecNotFoundError(fully_qualified_name, {}))

        return self._spec_cache.get(fully_qualified_name, None)
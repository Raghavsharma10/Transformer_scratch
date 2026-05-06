def get_store(self, fully_qualified_name: str) -> Optional['Store']:
        """
        Used to generate a store object from the given fully_qualified_name.
        :param fully_qualified_name: The fully qualified name of the store object needed.
        :return: An initialized store object
        """

        if fully_qualified_name not in self._store_cache:
            schema = self.get_schema_object(fully_qualified_name)
            if not schema:
                return None

            if Type.is_store_type(schema.type):
                self._store_cache[fully_qualified_name] = TypeLoader.load_item(schema.type)(schema)
            else:
                self.add_errors(
                    InvalidTypeError(fully_qualified_name, {}, ATTRIBUTE_TYPE,
                                     InvalidTypeError.Reason.INCORRECT_BASE, schema.type,
                                     InvalidTypeError.BaseTypes.STORE))

        return self._store_cache.get(fully_qualified_name, None)
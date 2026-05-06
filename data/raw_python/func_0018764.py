def add_errors(self, *errors: Union[BaseSchemaError, SchemaErrorCollection]) -> None:
        """ Adds errors to the error store for the schema """
        for error in errors:
            self._error_cache.add(error)
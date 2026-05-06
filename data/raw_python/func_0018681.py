def add_errors(self, *errors: Union[BaseSchemaError, List[BaseSchemaError]]) -> None:
        """ Adds errors to the error repository in schema loader """
        self.schema_loader.add_errors(*errors)
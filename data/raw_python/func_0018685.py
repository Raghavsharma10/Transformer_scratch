def validate_schema_spec(self) -> None:
        """ Contains the validation routines that are to be executed as part of initialization by subclasses.
        When this method is being extended, the first line should always be: ```super().validate_schema_spec()``` """
        self.add_errors(
            validate_empty_attributes(self.fully_qualified_name, self._spec, *self._spec.keys()))
        self.add_errors(
            validate_python_identifier_attributes(self.fully_qualified_name, self._spec,
                                                  self.ATTRIBUTE_NAME))
def validate_required_attributes(self, *attributes: str) -> None:
        """ Validates that the schema contains a series of required attributes """
        self.add_errors(
            validate_required_attributes(self.fully_qualified_name, self._spec, *attributes))
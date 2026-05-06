def validate_enum_attribute(self, attribute: str,
                                candidates: Set[Union[str, int, float]]) -> None:
        """ Validates that the attribute value is among the candidates """
        self.add_errors(
            validate_enum_attribute(self.fully_qualified_name, self._spec, attribute, candidates))
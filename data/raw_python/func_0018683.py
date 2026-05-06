def validate_number_attribute(self,
                                  attribute: str,
                                  value_type: Union[Type[int], Type[float]] = int,
                                  minimum: Optional[Union[int, float]] = None,
                                  maximum: Optional[Union[int, float]] = None) -> None:
        """ Validates that the attribute contains a numeric value within boundaries if specified """
        self.add_errors(
            validate_number_attribute(self.fully_qualified_name, self._spec, attribute, value_type,
                                      minimum, maximum))
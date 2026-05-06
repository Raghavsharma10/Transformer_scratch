def validate_number_attribute(
        fully_qualified_name: str,
        spec: Dict[str, Any],
        attribute: str,
        value_type: Union[Type[int], Type[float]] = int,
        minimum: Optional[Union[int, float]] = None,
        maximum: Optional[Union[int, float]] = None) -> Optional[InvalidNumberError]:
    """ Validates to ensure that the value is a number of the specified type, and lies with the specified range """

    if attribute not in spec:
        return

    try:
        value = value_type(spec[attribute])
        if (minimum is not None and value < minimum) or (maximum is not None and value > maximum):
            raise None
    except:
        return InvalidNumberError(fully_qualified_name, spec, attribute, value_type, minimum,
                                  maximum)
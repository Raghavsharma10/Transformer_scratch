def validate_enum_attribute(fully_qualified_name: str, spec: Dict[str, Any], attribute: str,
                            candidates: Set[Union[str, int, float]]) -> Optional[InvalidValueError]:
    """ Validates to ensure that the value of an attribute lies within an allowed set of candidates """

    if attribute not in spec:
        return

    if spec[attribute] not in candidates:
        return InvalidValueError(fully_qualified_name, spec, attribute, candidates)
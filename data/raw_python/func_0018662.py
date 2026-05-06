def validate_empty_attributes(fully_qualified_name: str, spec: Dict[str, Any],
                              *attributes: str) -> List[EmptyAttributeError]:
    """ Validates to ensure that a set of attributes do not contain empty values """
    return [
        EmptyAttributeError(fully_qualified_name, spec, attribute)
        for attribute in attributes
        if not spec.get(attribute, None)
    ]
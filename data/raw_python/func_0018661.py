def validate_required_attributes(fully_qualified_name: str, spec: Dict[str, Any],
                                 *attributes: str) -> List[RequiredAttributeError]:
    """ Validates to ensure that a set of attributes are present in spec """
    return [
        RequiredAttributeError(fully_qualified_name, spec, attribute)
        for attribute in attributes
        if attribute not in spec
    ]
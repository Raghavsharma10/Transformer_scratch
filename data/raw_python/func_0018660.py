def validate_python_identifier_attributes(fully_qualified_name: str, spec: Dict[str, Any],
                                          *attributes: str) -> List[InvalidIdentifierError]:
    """ Validates a set of attributes as identifiers in a spec """
    errors: List[InvalidIdentifierError] = []

    checks: List[Tuple[Callable, InvalidIdentifierError.Reason]] = [
        (lambda x: x.startswith('_'), InvalidIdentifierError.Reason.STARTS_WITH_UNDERSCORE),
        (lambda x: x.startswith('run_'), InvalidIdentifierError.Reason.STARTS_WITH_RUN),
        (lambda x: not x.isidentifier(), InvalidIdentifierError.Reason.INVALID_PYTHON_IDENTIFIER),
    ]

    for attribute in attributes:
        if attribute not in spec or spec.get(ATTRIBUTE_INTERNAL, False):
            continue

        for check in checks:
            if check[0](spec[attribute]):
                errors.append(
                    InvalidIdentifierError(fully_qualified_name, spec, attribute, check[1]))
                break

    return errors
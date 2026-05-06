def generate_message(contract: Contract, condition_kwargs: Mapping[str, Any]) -> str:
    """Generate the message upon contract violation."""
    # pylint: disable=protected-access
    parts = []  # type: List[str]

    if contract.location is not None:
        parts.append("{}:\n".format(contract.location))

    if contract.description is not None:
        parts.append("{}: ".format(contract.description))

    lambda_inspection = None  # type: Optional[ConditionLambdaInspection]
    if not _is_lambda(a_function=contract.condition):
        condition_text = contract.condition.__name__
    else:
        # We need to extract the source code corresponding to the decorator since inspect.getsource() is broken with
        # lambdas.

        # Find the line corresponding to the condition lambda
        lines, condition_lineno = inspect.findsource(contract.condition)
        filename = inspect.getsourcefile(contract.condition)

        decorator_inspection = inspect_decorator(lines=lines, lineno=condition_lineno, filename=filename)
        lambda_inspection = find_lambda_condition(decorator_inspection=decorator_inspection)

        assert lambda_inspection is not None, \
            "Expected lambda_inspection to be non-None if _is_lambda is True on: {}".format(contract.condition)

        condition_text = lambda_inspection.text

    parts.append(condition_text)

    repr_vals = repr_values(
        condition=contract.condition,
        lambda_inspection=lambda_inspection,
        condition_kwargs=condition_kwargs,
        a_repr=contract._a_repr)

    if len(repr_vals) == 0:
        # Do not append anything since no value could be represented as a string.
        # This could appear in case we have, for example, a generator expression as the return value of a lambda.
        pass

    elif len(repr_vals) == 1:
        parts.append(': ')
        parts.append(repr_vals[0])
    else:
        parts.append(':\n')
        parts.append('\n'.join(repr_vals))

    msg = "".join(parts)
    return msg
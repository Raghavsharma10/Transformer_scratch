def repr_values(condition: Callable[..., bool], lambda_inspection: Optional[ConditionLambdaInspection],
                condition_kwargs: Mapping[str, Any], a_repr: reprlib.Repr) -> List[str]:
    # pylint: disable=too-many-locals
    """
    Represent function arguments and frame values in the error message on contract breach.

    :param condition: condition function of the contract
    :param lambda_inspection:
        inspected lambda AST node corresponding to the condition function (None if the condition was not given as a
        lambda function)
    :param condition_kwargs: condition arguments
    :param a_repr: representation instance that defines how the values are represented.
    :return: list of value representations
    """
    if _is_lambda(a_function=condition):
        assert lambda_inspection is not None, "Expected a lambda inspection when given a condition as a lambda function"
    else:
        assert lambda_inspection is None, "Expected no lambda inspection in a condition given as a non-lambda function"

    reprs = dict()  # type: MutableMapping[str, Any]

    if lambda_inspection is not None:
        # Collect the variable lookup of the condition function:
        variable_lookup = []  # type: List[Mapping[str, Any]]

        # Add condition arguments to the lookup
        variable_lookup.append(condition_kwargs)

        # Add closure to the lookup
        closure_dict = dict()  # type: Dict[str, Any]

        if condition.__closure__ is not None:  # type: ignore
            closure_cells = condition.__closure__  # type: ignore
            freevars = condition.__code__.co_freevars

            assert len(closure_cells) == len(freevars), \
                "Number of closure cells of a condition function ({}) == number of free vars ({})".format(
                    len(closure_cells), len(freevars))

            for cell, freevar in zip(closure_cells, freevars):
                closure_dict[freevar] = cell.cell_contents

        variable_lookup.append(closure_dict)

        # Add globals to the lookup
        if condition.__globals__ is not None:  # type: ignore
            variable_lookup.append(condition.__globals__)  # type: ignore

        # pylint: disable=protected-access
        recompute_visitor = icontract._recompute.Visitor(variable_lookup=variable_lookup)

        recompute_visitor.visit(node=lambda_inspection.node.body)
        recomputed_values = recompute_visitor.recomputed_values

        repr_visitor = Visitor(
            recomputed_values=recomputed_values, variable_lookup=variable_lookup, atok=lambda_inspection.atok)
        repr_visitor.visit(node=lambda_inspection.node.body)

        reprs = repr_visitor.reprs
    else:
        for key, val in condition_kwargs.items():
            if _representable(value=val):
                reprs[key] = val

    parts = []  # type: List[str]
    for key in sorted(reprs.keys()):
        parts.append('{} was {}'.format(key, a_repr.repr(reprs[key])))

    return parts
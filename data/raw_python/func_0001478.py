def find_lambda_condition(decorator_inspection: DecoratorInspection) -> Optional[ConditionLambdaInspection]:
    """
    Inspect the decorator and extract the condition as lambda.

    If the condition is not given as a lambda function, return None.
    """
    call_node = decorator_inspection.node

    lambda_node = None  # type: Optional[ast.Lambda]

    if len(call_node.args) > 0:
        assert isinstance(call_node.args[0], ast.Lambda), \
            ("Expected the first argument to the decorator to be a condition as lambda AST node, "
             "but got: {}").format(type(call_node.args[0]))

        lambda_node = call_node.args[0]

    elif len(call_node.keywords) > 0:
        for keyword in call_node.keywords:
            if keyword.arg == "condition":
                assert isinstance(keyword.value, ast.Lambda), \
                    "Expected lambda node as value of the 'condition' argument to the decorator."

                lambda_node = keyword.value
                break

        assert lambda_node is not None, "Expected to find a keyword AST node with 'condition' arg, but found none"
    else:
        raise AssertionError(
            "Expected a call AST node of a decorator to have either args or keywords, but got: {}".format(
                ast.dump(call_node)))

    return ConditionLambdaInspection(atok=decorator_inspection.atok, node=lambda_node)
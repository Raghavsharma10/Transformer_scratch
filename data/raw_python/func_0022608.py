def infer(query, replacements=None, root_type=None,
          libs=("stdcore", "stdmath")):
    """Determine the type of the query's output without actually running it.

    Arguments:
        query: A query object or string with the query.
        replacements: Built-time parameters to the query, either as dict or as
            an array (for positional interpolation).
        root_type: The types of variables to be supplied to the query inference.
        libs: What standard libraries should be taken into account for the
            inference.

    Returns:
        The type of the query's output, if it can be determined. If undecidable,
        returns efilter.protocol.AnyType.

        NOTE: The inference returns the type of a row in the results, not of the
        actual Python object returned by 'apply'. For example, if a query
        returns multiple rows, each one of which is an integer, the type of the
        output is considered to be int, not a collection of rows.

    Examples:
        infer("5 + 5") # -> INumber

        infer("SELECT * FROM people WHERE age > 10") # -> AnyType

        # If root_type implements the IStructured reflection API:
        infer("SELECT * FROM people WHERE age > 10", root_type=...) # -> dict
    """
    # Always make the scope stack start with stdcore.
    if root_type:
        type_scope = scope.ScopeStack(std_core.MODULE, root_type)
    else:
        type_scope = scope.ScopeStack(std_core.MODULE)

    stdcore_included = False
    for lib in libs:
        if lib == "stdcore":
            stdcore_included = True
            continue

        module = std_core.LibraryModule.ALL_MODULES.get(lib)
        if not module:
            raise TypeError("No standard library module %r." % lib)

        type_scope = scope.ScopeStack(module, type_scope)

    if not stdcore_included:
        raise TypeError("'stdcore' must always be included.")

    query = q.Query(query, params=replacements)
    return infer_type.infer_type(query, type_scope)
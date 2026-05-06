def apply(query, replacements=None, vars=None, allow_io=False,
          libs=("stdcore", "stdmath")):
    """Run 'query' on 'vars' and return the result(s).

    Arguments:
        query: A query object or string with the query.
        replacements: Built-time parameters to the query, either as dict or
            as an array (for positional interpolation).
        vars: The variables to be supplied to the query solver.
        allow_io: (Default: False) Include 'stdio' and allow IO functions.
        libs: Iterable of library modules to include, given as strings.
            Default: ('stdcore', 'stdmath')
            For full list of bundled libraries, see efilter.stdlib.

            Note: 'stdcore' must always be included.

            WARNING: Including 'stdio' must be done in conjunction with
                'allow_io'. This is to make enabling IO explicit. 'allow_io'
                implies that 'stdio' should be included and so adding it to
                libs is actually not required.

    Notes on IO: If allow_io is set to True then 'stdio' will be included and
    the EFILTER query will be allowed to read files from disk. Use this with
    caution.

        If the query returns a lazily-evaluated result that depends on reading
        from a file (for example, filtering a CSV file) then the file
        descriptor will remain open until the returned result is deallocated.
        The caller is responsible for releasing the result when it's no longer
        needed.

    Returns:
        The result of evaluating the query. The type of the output will depend
        on the query, and can be predicted using 'infer' (provided reflection
        callbacks are implemented). In the common case of a SELECT query the
        return value will be an iterable of filtered data (actually an object
        implementing IRepeated, as well as __iter__.)

    A word on cardinality of the return value:
        Types in EFILTER always refer to a scalar. If apply returns more than
        one value, the type returned by 'infer' will refer to the type of
        the value inside the returned container.

        If you're unsure whether your query returns one or more values (rows),
        use the 'getvalues' function.

    Raises:
        efilter.errors.EfilterError if there are issues with the query.

    Examples:
        apply("5 + 5") # -> 10

        apply("SELECT * FROM people WHERE age > 10",
              vars={"people":({"age": 10, "name": "Bob"},
                              {"age": 20, "name": "Alice"},
                              {"age": 30, "name": "Eve"}))

        # This will replace the question mark (?) with the string "Bob" in a
        # safe manner, preventing SQL injection.
        apply("SELECT * FROM people WHERE name = ?", replacements=["Bob"], ...)
    """
    if vars is None:
        vars = {}

    if allow_io:
        libs = list(libs)
        libs.append("stdio")

    query = q.Query(query, params=replacements)

    stdcore_included = False
    for lib in libs:
        if lib == "stdcore":
            stdcore_included = True
            # 'solve' always includes this automatically - we don't have a say
            # in the matter.
            continue

        if lib == "stdio" and not allow_io:
            raise ValueError("Attempting to include 'stdio' but IO not "
                             "enabled. Pass allow_io=True.")

        module = std_core.LibraryModule.ALL_MODULES.get(lib)
        if not lib:
            raise ValueError("There is no standard library module %r." % lib)
        vars = scope.ScopeStack(module, vars)

    if not stdcore_included:
        raise ValueError("EFILTER cannot work without standard lib 'stdcore'.")

    results = solve.solve(query, vars).value

    return results
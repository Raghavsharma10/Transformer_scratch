def extract_schemas_from_source(source, filename='<unknown>'):
    """Extract schemas from 'source'.

    The 'source' parameter must be a string, and should be valid python
    source.

    If 'source' is not valid python source, a SyntaxError will be raised.

    :returns: a list of ViewSchema objects.
    """
    # Track which acceptable services have been configured.
    acceptable_services = set()
    # Track which acceptable views have been configured:
    acceptable_views = {}
    schemas_found = []
    ast_tree = ast.parse(source, filename)
    simple_names = _get_simple_assignments(ast_tree)

    assigns = [n for n in ast_tree.body if isinstance(n, ast.Assign)]
    call_assigns = [n for n in assigns if isinstance(n.value, ast.Call)]

    # We need to extract the AcceptableService-related views. We parse the
    # assignations twice: The first time to extract the AcceptableService
    # instances, the second to extract the views created on those services.
    for assign in call_assigns:
        if isinstance(assign.value.func, ast.Attribute):
            continue
        if assign.value.func.id == 'AcceptableService':
            for target in assign.targets:
                acceptable_services.add(target.id)

    for assign in call_assigns:
        # only consider calls which are attribute accesses, AND
        # calls where the object being accessed is in acceptable_services, AND
        # calls where the attribute being accessed is the 'api' method.
        if isinstance(assign.value.func, ast.Attribute) and \
           assign.value.func.value.id in acceptable_services and \
           assign.value.func.attr == 'api':
            # this is a view. We need to extract the url and methods specified.
            # they may be specified positionally or via a keyword.
            url = None
            name = None
            # methods has a default value:
            methods = ['GET']

            # This is a view - the URL is the first positional argument:
            args = assign.value.args
            if len(args) >= 1:
                url = ast.literal_eval(args[0])
            if len(args) >= 2:
                name = ast.literal_eval(args[1])
            kwargs = assign.value.keywords
            for kwarg in kwargs:
                if kwarg.arg == 'url':
                    url = ast.literal_eval(kwarg.value)
                if kwarg.arg == 'methods':
                    methods = ast.literal_eval(kwarg.value)
                if kwarg.arg == 'view_name':
                    name = ast.literal_eval(kwarg.value)
            if url and name:
                for target in assign.targets:
                    acceptable_views[target.id] = {
                        'url': url,
                        'name': name,
                        'methods': methods,
                    }

    # iterate over all functions, attempting to find the views.
    functions = [n for n in ast_tree.body if isinstance(n, ast.FunctionDef)]
    for function in functions:
        input_schema = None
        output_schema = None
        doc = ast.get_docstring(function)
        api_options_list = []
        for decorator in function.decorator_list:
            if not isinstance(decorator, ast.Call):
                continue
            if isinstance(decorator.func, ast.Attribute):
                decorator_name = decorator.func.value.id
                # extract version this view was introduced at, which can be
                # specified as an arg or a kwarg:
                version = None
                for kwarg in decorator.keywords:
                    if kwarg.arg == 'introduced_at':
                        version = ast.literal_eval(kwarg.value)
                        break
                if len(decorator.args) == 1:
                    version = ast.literal_eval(decorator.args[0])

                if decorator_name in acceptable_views:
                    api_options = acceptable_views[decorator_name]
                    api_options['version'] = version
                    api_options_list.append(api_options)
            else:
                decorator_name = decorator.func.id
                if decorator_name == 'validate_body':
                    _SimpleNamesResolver(simple_names).visit(decorator.args[0])
                    input_schema = ast.literal_eval(decorator.args[0])
                if decorator_name == 'validate_output':
                    _SimpleNamesResolver(simple_names).visit(decorator.args[0])
                    output_schema = ast.literal_eval(decorator.args[0])
        for api_options in api_options_list:
            schema = ViewSchema(
                    view_name=api_options['name'],
                    version=api_options['version'],
                    input_schema=input_schema,
                    output_schema=output_schema,
                    methods=api_options['methods'],
                    url=api_options['url'],
                    doc=doc,
                )
            schemas_found.append(schema)
    return schemas_found
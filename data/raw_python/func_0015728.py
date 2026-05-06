def generate_dummy_callable(info, func_name, method=False,
                            signal_owner_type=None):
    """Takes a GICallableInfo and generates a dummy callback function which
    just raises but has a correct docstring. They are mainly accessible for
    documentation, so the API reference can reference a real thing.

    func_name can be different than info.name because vfuncs, for example,
    get prefixed with 'do_' when exposed in Python.
    """

    assert isinstance(info, GICallableInfo)

    # FIXME: handle out args and trailing user_data ?

    arg_infos = list(info.get_args())
    arg_types = [a.get_type() for a in arg_infos]
    return_type = info.get_return_type()

    # the null backend is good enough here
    backend = get_backend("null")()

    args = []
    for arg_info, arg_type in zip(arg_infos, arg_types):
        cls = get_argument_class(arg_type)
        name = escape_identifier(arg_info.name)
        name = escape_parameter(name)
        args.append(cls(name, args, backend, arg_info, arg_type))

    cls = get_return_class(return_type)
    return_value = cls(info, return_type, args, backend)

    for arg in args:
        arg.setup()

    return_value.setup()

    in_args = [a for a in args if not a.is_aux and a.in_var]

    # if the last in argument is a closure, make it a var-positional argument
    if in_args and in_args[-1].closure != -1:
        name = in_args[-1].in_var
        in_args[-1].in_var = "*" + name

    func_name = escape_identifier(func_name)
    docstring = build_docstring(func_name, args, return_value,
                                False, signal_owner_type)

    in_names = [a.in_var for a in in_args]

    var_fac = backend.var
    var_fac.add_blacklist(in_names)
    self_name = ""
    if method:
        self_name = var_fac.request_name("self")
        in_names.insert(0, self_name)

    main, var = backend.parse("""
def $func_name($func_args):
    '''$docstring'''

    raise NotImplementedError("This is just a dummy callback function")
""", func_args=", ".join(in_names), docstring=docstring, func_name=func_name)

    func = main.compile()[func_name]
    func._code = main
    func.__doc__ = docstring
    func.__module__ = info.namespace

    return func
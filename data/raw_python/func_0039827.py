def parse_arguments(filters, arguments, modern=False):
    """
    Return a dict of parameters.
    Take a list of filters and for each try to get the corresponding
    value in arguments or a default value. Then check that value's type.
    The @modern parameter indicates how the arguments should be
    interpreted. The old way is that you always specify a list and in
    the list you write the names of types as strings. I.e. instad of
    `str` you write `'str'`.
    The modern way allows you to specify arguments by real Python types
    and entering it as a list means you accept and expect it to be a list.
    For example, using the modern way:
        filters = [
            ("param1", "default", [str]),
            ("param2", None, int),
            ("param3", ["list", "of", 4, "values"], [str])
        ]
        arguments = {
            "param1": "value1",
            "unknown": 12345
        }
        =>
        {
            "param1": ["value1"],
            "param2": 0,
            "param3": ["list", "of", "4", "values"]
        }
    And an example for the old way:
        filters = [
            ("param1", "default", ["list", "str"]),
            ("param2", None, "int"),
            ("param3", ["list", "of", 4, "values"], ["list", "str"])
        ]
        arguments = {
            "param1": "value1",
            "unknown": 12345
        }
        =>
        {
            "param1": ["value1"],
            "param2": 0,
            "param3": ["list", "of", "4", "values"]
        }
    The reason for having the modern and the non-modern way is
    transition of legacy code. One day it will all be the modern way.
    """
    params = DotDict()

    for i in filters:
        count = len(i)
        param = None

        if count <= 1:
            param = arguments.get(i[0])
        else:
            param = arguments.get(i[0], i[1])

        # proceed and do the type checking
        if count >= 3:
            types = i[2]

            if modern:
                if isinstance(types, list) and param is not None:
                    assert len(types) == 1
                    if not isinstance(param, list):
                        param = [param]
                    param = [check_type(x, types[0]) for x in param]
                else:
                    param = check_type(param, types)
            else:
                if not isinstance(types, list):
                    types = [types]

                for t in reversed(types):
                    if t == "list" and not isinstance(param, list):
                        if param is None or param == '':
                            param = []
                        else:
                            param = [param]
                    elif t == "list" and isinstance(param, list):
                        continue
                    elif isinstance(param, list) and "list" not in types:
                        param = " ".join(param)
                        param = check_type(param, t)
                    elif isinstance(param, list):
                        param = [check_type(x, t) for x in param]
                    else:
                        param = check_type(param, t)

        params[i[0]] = param
    return params
def generate_function(info, method=False):
    """Creates a Python callable for a GIFunctionInfo instance"""

    assert isinstance(info, GIFunctionInfo)

    arg_infos = list(info.get_args())
    arg_types = [a.get_type() for a in arg_infos]
    return_type = info.get_return_type()

    func = None
    messages = []
    for backend in list_backends():
        instance = backend()
        try:
            func = _generate_function(instance, info, arg_infos, arg_types,
                                      return_type, method)
        except NotImplementedError:
            messages.append("%s: %s" % (backend.NAME, traceback.format_exc()))
        else:
            break

    if func:
        return func

    raise NotImplementedError("\n".join(messages))
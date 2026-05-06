def get_function_by_name(full_name):
    """
    RETURN FUNCTION
    """

    # IMPORT MODULE FOR HANDLER
    path = full_name.split(".")
    function_name = path[-1]
    path = ".".join(path[:-1])
    constructor = None
    try:
        temp = __import__(path, globals(), locals(), [function_name], -1)
        output = object.__getattribute__(temp, function_name)
        return output
    except Exception as e:
        Log.error("Can not find function {{name}}",  name= full_name, cause=e)
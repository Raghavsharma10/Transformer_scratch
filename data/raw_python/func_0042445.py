def index_to_coordinate(dims):
    """
    RETURN A FUNCTION THAT WILL TAKE AN INDEX, AND MAP IT TO A coordinate IN dims

    :param dims: TUPLE WITH NUMBER OF POINTS IN EACH DIMENSION
    :return: FUNCTION
    """
    _ = divmod  # SO WE KEEP THE IMPORT

    num_dims = len(dims)
    if num_dims == 0:
        return _zero_dim

    prod = [1] * num_dims
    acc = 1
    domain = range(0, num_dims)
    for i in reversed(domain):
        prod[i] = acc
        acc *= dims[i]

    commands = []
    coords = []
    for i in domain:
        if i == num_dims - 1:
            commands.append("\tc" + text_type(i) + " = index")
        else:
            commands.append("\tc" + text_type(i) + ", index = divmod(index, " + text_type(prod[i]) + ")")
        coords.append("c" + text_type(i))
    output = None
    if num_dims == 1:
        code = (
            "def output(index):\n" +
            "\n".join(commands) + "\n" +
            "\treturn " + coords[0] + ","
        )
    else:
        code = (
            "def output(index):\n" +
            "\n".join(commands) + "\n" +
            "\treturn " + ", ".join(coords)
        )

    exec(code)
    return output
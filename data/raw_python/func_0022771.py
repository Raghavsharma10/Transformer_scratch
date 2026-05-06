def as_es2_command(command):
    """ Modify a desktop command so it works on es2.
    """

    if command[0] == 'FUNC':
        return (command[0], re.sub(r'^gl([A-Z])',
                lambda m: m.group(1).lower(), command[1])) + command[2:]
    if command[0] == 'SHADERS':
        return command[:2] + convert_shaders('es2', command[2:])
    if command[0] == 'UNIFORM':
        return command[:-1] + (command[-1].tolist(),)
    return command
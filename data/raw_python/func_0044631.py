def execute(args):
    """
    Call vswhere with the given arguments and return an array of results.

    `args` is a list of command line arguments to pass to vswhere.

    If the argument list contains '-property', this returns an array with the
    property value for each result. Otherwise, this returns an array of
    dictionaries containing the results.
    """
    is_property = '-property' in args

    args = [get_vswhere_path(), '-utf8'] + args

    if not is_property:
        args.extend(['-format', 'json'])

    output = subprocess.check_output(args)

    if is_property:
        return output.decode('utf-8').splitlines()
    else:
        return json.loads(output, encoding='utf-8')
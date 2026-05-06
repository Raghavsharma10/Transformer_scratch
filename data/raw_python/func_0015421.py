def run_in_syspy(f):
    """
    Decorator to run a function in the system python

    :param f:
    :return:
    """
    fname = f.__name__
    code_lines = inspect.getsource(f).splitlines()

    code = dedent("\n".join(code_lines[1:]))  # strip this decorator

    # add call to the function and print it's result
    code += dedent("""\n
        import sys
        args = sys.argv[1:]
        result = {fname}(*args)
        print("%r" % result)
    """).format(fname=fname)

    env = os.environ
    python = findsyspy()
    logger.debug("Create function for system python\n%s" % code)

    def call_f(*args):
        cmd = [python, '-c', code] + list(args)
        output = subprocess.check_output(cmd, env=env).decode('utf-8')
        result = ast.literal_eval(output)
        return result

    return call_f
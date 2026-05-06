def cwd_decorator(func):
    """
    decorator to change cwd to directory containing rst for this function
    """

    def wrapper(*args, **kw):
        cur_dir = os.getcwd()
        found = False
        for arg in sys.argv:
            if arg.endswith(".rst"):
                found = arg
                break

        if found:
            directory = os.path.dirname(found)
            if directory:
                os.chdir(directory)
        data = func(*args, **kw)
        os.chdir(cur_dir)
        return data

    return wrapper
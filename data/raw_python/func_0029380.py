def status(*args, **kwargs):
    """Prints a status message at the start and finish of an associated
    function. Can be used as a function decorator or as a function that accepts
    another function as the first parameter.

    **Params**:

    The following parameters are available when used as a decorator:

      - msg (str) [args] - Message to print at start of `func`.

    The following parameters are available when used as a function:

      - msg (str) [args] - Message to print at start of `func`.
      - func (func) - Function to call. First `args` if using `status()` as a
        function. Automatically provided if using `status()` as a decorator.
      - fargs (list) - List of `args` passed to `func`.
      - fkrgs (dict) - Dictionary of `kwargs` passed to `func`.
      - fin (str) [kwargs] - Message to print when `func` finishes.

    **Examples**:
    ::
        @qprompt.status("Something is happening...")
        def do_something(a):
            time.sleep(a)

        do_something()
        # [!] Something is happening... DONE.

        qprompt.status("Doing a thing...", myfunc, [arg1], {krgk: krgv})
        # [!] Doing a thing... DONE.
    """
    def decor(func):
        @wraps(func)
        def wrapper(*args, **krgs):
            echo("[!] " + msg, end=" ", flush=True)
            result = func(*args, **krgs)
            echo(fin, flush=True)
            return result
        return wrapper
    fin = kwargs.pop('fin', "DONE.")
    args = list(args)
    if len(args) > 1 and callable(args[1]):
        msg = args.pop(0)
        func = args.pop(0)
        try: fargs = args.pop(0)
        except: fargs = []
        try: fkrgs = args.pop(0)
        except: fkrgs = {}
        return decor(func)(*fargs, **fkrgs)
    msg = args.pop(0)
    return decor
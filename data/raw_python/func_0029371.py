def run_func(entry):
    """Runs the function associated with the given MenuEntry."""
    if entry.func:
        if entry.args and entry.krgs:
            return entry.func(*entry.args, **entry.krgs)
        if entry.args:
            return entry.func(*entry.args)
        if entry.krgs:
            return entry.func(**entry.krgs)
        return entry.func()
def run_dot(dot):
    """Converts a graph in DOT format into an IPython displayable object."""
    global impl
    if impl is None:
        impl = guess_impl()
    if impl == "dot":
        return run_dot_dot(dot)
    elif impl == "js":
        return run_dot_js(dot)
    else:
        raise ValueError("unknown implementation {}".format(impl))
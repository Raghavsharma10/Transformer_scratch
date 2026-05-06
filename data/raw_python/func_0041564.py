def requires_git(func: Callable) -> Callable:
    """
    Decorator to ensure `git` is accessible before calling a function.
    :param func: the function to wrap
    :return: the wrapped function
    """
    def decorated(*args, **kwargs):
        try:
            run([GIT_COMMAND, "--version"])
        except RunException as e:
            raise RuntimeError("`git` does not appear to be working") from e
        return func(*args, **kwargs)

    return decorated
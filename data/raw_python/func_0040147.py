def requires_subrepo(func: Callable) -> Callable:
    """
    Decorator that requires the `git subrepo` command to be accessible before calling the given function.
    :param func: the function to wrap
    :return: the wrapped function
    """
    def decorated(*args, **kwargs):
        try:
            run([GIT_COMMAND, _GIT_SUBREPO_COMMAND, "--version"])
        except RunException as e:
            raise RuntimeError("`git subrepo` does not appear to be working") from e
        return func(*args, **kwargs)

    return decorated
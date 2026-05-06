def transition(trname='', field='', check=None, before=None, after=None):
    """Decorator to declare a function as a transition implementation."""
    if is_callable(trname):
        raise ValueError(
            "The @transition decorator should be called as "
            "@transition(['transition_name'], **kwargs)")
    if check or before or after:
        warnings.warn(
            "The use of check=, before= and after= in @transition decorators is "
            "deprecated in favor of @transition_check, @before_transition and "
            "@after_transition decorators.",
            DeprecationWarning,
            stacklevel=2)
    return TransitionWrapper(trname, field=field, check=check, before=before, after=after)
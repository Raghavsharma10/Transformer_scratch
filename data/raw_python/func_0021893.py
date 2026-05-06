def use_defaults(func):
    """
    Decorator for functions that should automatically fall back to the Cohort-default filter_fn and
    normalized_per_mb if not specified.
    """
    @wraps(func)
    def wrapper(row, cohort, filter_fn=None, normalized_per_mb=None, **kwargs):
        filter_fn = first_not_none_param([filter_fn, cohort.filter_fn], no_filter)
        normalized_per_mb = first_not_none_param([normalized_per_mb, cohort.normalized_per_mb], False)
        return func(row=row,
                    cohort=cohort,
                    filter_fn=filter_fn,
                    normalized_per_mb=normalized_per_mb,
                    **kwargs)
    return wrapper
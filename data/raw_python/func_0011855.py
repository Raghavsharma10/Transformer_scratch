def rate_limited(max_per_hour: int, *args: Any) -> Callable[..., Any]:
    """Rate limit a function."""
    return util.rate_limited(max_per_hour, *args)
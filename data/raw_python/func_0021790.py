def raise_for_status(
    status: int, headers: MutableMapping, data: MutableMapping
) -> None:
    """
    Check request response status

    Args:
        status: Response status
        headers: Response headers
        data: Response data

    Raises:
        :class:`slack.exceptions.RateLimited`: For 429 status code
        :class:`slack.exceptions:HTTPException`:
    """
    if status != 200:
        if status == 429:

            if isinstance(data, str):
                error = data
            else:
                error = data.get("error", "ratelimited")

            try:
                retry_after = int(headers.get("Retry-After", 1))
            except ValueError:
                retry_after = 1
            raise exceptions.RateLimited(retry_after, error, status, headers, data)
        else:
            raise exceptions.HTTPException(status, headers, data)
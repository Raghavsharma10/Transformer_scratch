def check_url(url):
    """Check whether the given URL is dead or alive.

    Returns a dict with four keys:

        "url": The URL that was checked (string)
        "alive": Whether the URL was working, True or False
        "status": The HTTP status code of the response from the URL,
            e.g. 200, 401, 500 (int)
        "reason": The reason for the success or failure of the check,
            e.g. "OK", "Unauthorized", "Internal Server Error" (string)

    The "status" may be None if we did not get a valid HTTP response,
    e.g. in the event of a timeout, DNS failure or invalid HTTP response.

    The "reason" will always be a string, but may be a requests library
    exception string rather than an HTTP reason string if we did not get a valid
    HTTP response.

    """
    result = {"url": url}
    try:
        response = requests.get(url)
        result["status"] = response.status_code
        result["reason"] = response.reason
        response.raise_for_status()  # Raise if status_code is not OK.
        result["alive"] = True
    except AttributeError as err:
        if err.message == "'NoneType' object has no attribute 'encode'":
            # requests seems to throw these for some invalid URLs.
            result["alive"] = False
            result["reason"] = "Invalid URL"
            result["status"] = None
        else:
            raise
    except requests.exceptions.RequestException as err:
        result["alive"] = False
        if "reason" not in result:
            result["reason"] = str(err)
        if "status" not in result:
            # This can happen if the response is invalid HTTP, if we get a DNS
            # failure, or a timeout, etc.
            result["status"] = None

    # We should always have these four fields in the result.
    assert "url" in result
    assert result.get("alive") in (True, False)
    assert "status" in result
    assert "reason" in result

    return result
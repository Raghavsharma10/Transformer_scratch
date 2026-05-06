def from_response(response, method, url):
    """Returns an instance of :class:`HttpError` or subclass based on response.

    :param response: instance of `requests.Response` class
    :param method: HTTP method used for request
    :param url: URL used for request
    """

    req_id = response.headers.get("x-openstack-request-id")
    # NOTE(hdd) true for older versions of nova and cinder
    if not req_id:
        req_id = response.headers.get("x-compute-request-id")
    kwargs = {
        "http_status": response.status_code,
        "response": response,
        "method": method,
        "url": url,
        "request_id": req_id,
    }
    if "retry-after" in response.headers:
        kwargs["retry_after"] = response.headers["retry-after"]

    content_type = response.headers.get("Content-Type", "")
    if content_type.startswith("application/json"):
        try:
            body = response.json()
        except ValueError:
            pass
        else:
            if isinstance(body, dict):
                error = body.get(list(body)[0])
                if isinstance(error, dict):
                    kwargs["message"] = (error.get("message") or
                                         error.get("faultstring"))
                    kwargs["details"] = (error.get("details") or
                                         six.text_type(body))
    elif content_type.startswith("text/"):
        kwargs["details"] = response.text

    try:
        cls = _code_map[response.status_code]
    except KeyError:
        if 500 <= response.status_code < 600:
            cls = HttpServerError
        elif 400 <= response.status_code < 500:
            cls = HTTPClientError
        else:
            cls = HttpError
    return cls(**kwargs)
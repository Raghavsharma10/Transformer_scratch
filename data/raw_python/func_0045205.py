def _request_json(
    url,
    parameters=None,
    body=None,
    headers=None,
    cache=True,
    agent=None,
    reattempt=5,
):
    """ Queries a url for json data

    Note: Requests are cached using requests_cached for a week, this is done
    transparently by using the package's monkey patching
    """
    assert url
    content = None
    status = 500
    log.info("url: %s" % url)

    if isinstance(headers, dict):
        headers = _clean_dict(headers)
    else:
        headers = dict()
    if isinstance(parameters, dict):
        parameters = _d2l(_clean_dict(parameters))
    if body:
        method = "POST"
        headers["content-type"] = "application/json"
        headers["user-agent"] = _get_user_agent(agent)
        headers["content-length"] = ustr(len(body))
    else:
        method = "GET"
        headers["user-agent"] = _get_user_agent(agent)

    initial_cache_state = SESSION._is_cache_disabled  # yes, i'm a bad person
    try:
        SESSION._is_cache_disabled = not cache
        response = SESSION.request(
            url=url,
            params=parameters,
            json=body,
            headers=headers,
            method=method,
            timeout=1,
        )
        status = response.status_code
        content = response.json() if status // 100 == 2 else None
        cache = getattr(response, "from_cache", False)
    except RequestException as e:
        log.debug(e, exc_info=True)
        return _request_json(
            url, parameters, body, headers, cache, agent, reattempt - 1
        )
    except Exception as e:
        log.error(e, exc_info=True)
        if reattempt > 0:
            SESSION.cache.clear()
            return _request_json(
                url, parameters, body, headers, False, agent, 0
            )
    else:
        log.info("method: %s" % method)
        log.info("headers: %r" % headers)
        log.info("parameters: %r" % parameters)
        log.info("cache: %r" % cache)
        log.info("status: %d" % status)
        log.debug("content: %s" % content)
    finally:
        SESSION._is_cache_disabled = initial_cache_state

    return status, content
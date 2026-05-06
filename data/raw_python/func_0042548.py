def request(method, url, headers=None, zip=None, retry=None, **kwargs):
    """
    JUST LIKE requests.request() BUT WITH DEFAULT HEADERS AND FIXES
    DEMANDS data IS ONE OF:
    * A JSON-SERIALIZABLE STRUCTURE, OR
    * LIST OF JSON-SERIALIZABLE STRUCTURES, OR
    * None

    Parameters
     * zip - ZIP THE REQUEST BODY, IF BIG ENOUGH
     * json - JSON-SERIALIZABLE STRUCTURE
     * retry - {"times": x, "sleep": y} STRUCTURE

    THE BYTE_STRINGS (b"") ARE NECESSARY TO PREVENT httplib.py FROM **FREAKING OUT**
    IT APPEARS requests AND httplib.py SIMPLY CONCATENATE STRINGS BLINDLY, WHICH
    INCLUDES url AND headers
    """
    global _warning_sent
    global request_count

    if not _warning_sent and not default_headers:
        Log.warning(text_type(
            "The pyLibrary.env.http module was meant to add extra " +
            "default headers to all requests, specifically the 'Referer' " +
            "header with a URL to the project. Use the `pyLibrary.debug.constants.set()` " +
            "function to set `pyLibrary.env.http.default_headers`"
        ))
    _warning_sent = True

    if is_list(url):
        # TRY MANY URLS
        failures = []
        for remaining, u in jx.countdown(url):
            try:
                response = request(method, u, retry=retry, **kwargs)
                if mo_math.round(response.status_code, decimal=-2) not in [400, 500]:
                    return response
                if not remaining:
                    return response
            except Exception as e:
                e = Except.wrap(e)
                failures.append(e)
        Log.error(u"Tried {{num}} urls", num=len(url), cause=failures)

    if 'session' in kwargs:
        session = kwargs['session']
        del kwargs['session']
        sess = Null
    else:
        sess = session = sessions.Session()

    with closing(sess):
        if PY2 and is_text(url):
            # httplib.py WILL **FREAK OUT** IF IT SEES ANY UNICODE
            url = url.encode('ascii')

        try:
            set_default(kwargs, {"zip":zip, "retry": retry}, DEFAULTS)
            _to_ascii_dict(kwargs)

            # HEADERS
            headers = kwargs['headers'] = unwrap(set_default(headers, session.headers, default_headers))
            _to_ascii_dict(headers)
            del kwargs['headers']

            # RETRY
            retry = wrap(kwargs['retry'])
            if isinstance(retry, Number):
                retry = set_default({"times":retry}, DEFAULTS['retry'])
            if isinstance(retry.sleep, Duration):
                retry.sleep = retry.sleep.seconds
            del kwargs['retry']

            # JSON
            if 'json' in kwargs:
                kwargs['data'] = value2json(kwargs['json']).encode('utf8')
                del kwargs['json']

            # ZIP
            set_default(headers, {'Accept-Encoding': 'compress, gzip'})

            if kwargs['zip'] and len(coalesce(kwargs.get('data'))) > 1000:
                compressed = convert.bytes2zip(kwargs['data'])
                headers['content-encoding'] = 'gzip'
                kwargs['data'] = compressed
            del kwargs['zip']
        except Exception as e:
            Log.error(u"Request setup failure on {{url}}", url=url, cause=e)

        errors = []
        for r in range(retry.times):
            if r:
                Till(seconds=retry.sleep).wait()

            try:
                DEBUG and Log.note(u"http {{method|upper}} to {{url}}", method=method, url=text_type(url))
                request_count += 1
                return session.request(method=method, headers=headers, url=str(url), **kwargs)
            except Exception as e:
                e = Except.wrap(e)
                if retry['http'] and str(url).startswith("https://") and "EOF occurred in violation of protocol" in e:
                    url = URL("http://" + str(url)[8:])
                    Log.note("Changed {{url}} to http due to SSL EOF violation.", url=str(url))
                errors.append(e)

        if " Read timed out." in errors[0]:
            Log.error(u"Tried {{times}} times: Timeout failure (timeout was {{timeout}}", timeout=kwargs['timeout'], times=retry.times, cause=errors[0])
        else:
            Log.error(u"Tried {{times}} times: Request failure of {{url}}", url=url, times=retry.times, cause=errors[0])
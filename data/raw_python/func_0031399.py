def respond_json(ctx, data, code = None, headers = [], json_dumps_default = None, jsonp = None):
    """Return a JSON response.

    This function is optimized for JSON following
    `Google JSON Style Guide <http://google-styleguide.googlecode.com/svn/trunk/jsoncstyleguide.xml>`_, but will handle
    any JSON except for HTTP errors.
    """
    if isinstance(data, collections.Mapping):
        # Remove null properties as recommended by Google JSON Style Guide.
        data = type(data)(
            (name, value)
            for name, value in data.iteritems()
            if value is not None
            )
        error = data.get('error')
        if isinstance(error, collections.Mapping):
            error = data['error'] = type(error)(
                (name, value)
                for name, value in error.iteritems()
                if value is not None
                )
    else:
        error = None
    if jsonp:
        content_type = 'application/javascript; charset=utf-8'
    else:
        content_type = 'application/json; charset=utf-8'
    if error:
        code = code or error['code']
        assert isinstance(code, int)
        response = webob.exc.status_map[code](headers = headers)
        response.content_type = content_type
        if code == 204:  # No content
            return response
        if error.get('code') is None:
            error['code'] = code
        if error.get('message') is None:
            title = errors_title.get(code)
            title = ctx._(title) if title is not None else response.status
            error['message'] = title
    else:
        response = ctx.req.response
        response.content_type = content_type
        if code is not None:
            response.status = code
        response.headers.update(headers)
    # try:
    #     text = json.dumps(data, encoding = 'utf-8', ensure_ascii = False, indent = 2)
    # except UnicodeDecodeError:
    #     text = json.dumps(data, ensure_ascii = True, indent = 2)

    if json_dumps_default is None:
        text = json.dumps(data)
    else:
        text = json.dumps(data, default = json_dumps_default)
    text = unicode(text)
    if jsonp:
        text = u'{0}({1})'.format(jsonp, text)
    response.text = text
    return response
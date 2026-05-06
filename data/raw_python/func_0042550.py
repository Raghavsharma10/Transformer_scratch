def post_json(url, **kwargs):
    """
    ASSUME RESPONSE IN IN JSON
    """
    if 'json' in kwargs:
        kwargs['data'] = unicode2utf8(value2json(kwargs['json']))
        del kwargs['json']
    elif 'data' in kwargs:
        kwargs['data'] = unicode2utf8(value2json(kwargs['data']))
    else:
        Log.error(u"Expecting `json` parameter")
    response = post(url, **kwargs)
    details = json2value(utf82unicode(response.content))
    if response.status_code not in [200, 201, 202]:

        if "template" in details:
            Log.error(u"Bad response code {{code}}", code=response.status_code, cause=Except.wrap(details))
        else:
            Log.error(u"Bad response code {{code}}\n{{details}}", code=response.status_code, details=details)
    else:
        return details
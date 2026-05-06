def get_json(url, **kwargs):
    """
    ASSUME RESPONSE IN IN JSON
    """
    response = get(url, **kwargs)
    try:
        c = response.all_content
        return json2value(utf82unicode(c))
    except Exception as e:
        if mo_math.round(response.status_code, decimal=-2) in [400, 500]:
            Log.error(u"Bad GET response: {{code}}", code=response.status_code)
        else:
            Log.error(u"Good GET requests, but bad JSON", cause=e)
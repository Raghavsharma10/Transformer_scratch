def response_hook(response, **kwargs) -> XMLResponse:
        """ Change response enconding and replace it by a HTMLResponse. """
        response.encoding = DEFAULT_ENCODING
        return XMLResponse._from_response(response)
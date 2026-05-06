def _handle_response(response, command, id_xpath='./id', **kwargs):
    """ Initialize the corect Response object from the response string based on the API command type. """
    _response_switch = {
        'insert': ModifyResponse,
        'replace': ModifyResponse,
        'partial-replace': ModifyResponse,
        'update': ModifyResponse,
        'delete': ModifyResponse,
        'search-delete': SearchDeleteResponse,
        'reindex': Response,
        'backup': Response,
        'restore': Response,
        'clear': Response,
        'status': StatusResponse,
        'search': SearchResponse,
        'retrieve': ListResponse,
        'similar': ListResponse,
        'lookup': LookupResponse,
        'alternatives': AlternativesResponse,
        'list-words': WordsResponse,
        'list-first': ListResponse,
        'list-last': ListResponse,
        'retrieve-last': ListResponse,
        'retrieve-first': ListResponse,
        'show-history': None,
        'list-paths': ListPathsResponse,
        'list-facets': ListFacetsResponse}
    try:
        request_class = _response_switch[command]
    except KeyError:
        request_class = Response
    return request_class(response, id_xpath, **kwargs)
def headers_to_include_from_request(curr_request):
    '''
        Define headers that needs to be included from the current request.
    '''
    return {
        h: v for h, v in curr_request.META.items() if h in _settings.HEADERS_TO_INCLUDE}
def make_response(response):
    """Make response tuple

    Potential features to be added
      - Parameters validation
    """
    if isinstance(response, unicode) or \
            isinstance(response, str):
        response = (response, 'text/html')

    return response
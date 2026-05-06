def handle_error(r, expected_code):
    """
    Helper function to match reponse of a request to the expected status
    code

    :param r: This field is the response of request.
    :param expected_code: This field is the expected status code for the
        function.
    """
    code = r.status_code
    if code != expected_code:
        info = 'API response status code {}'.format(code)
        try:
            if 'detail' in r.json():
                info = info + ": {}".format(r.json()['detail'])
            elif 'metadata' in r.json():
                info = info + ": {}".format(r.json()['metadata'])
        except json.decoder.JSONDecodeError:
            info = info + ":\n{}".format(r.content)
        raise Exception(info)
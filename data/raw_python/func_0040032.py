def success(headers = None, data = ''):
    """ Generate success JSON to send to client """
    passed_headers = {} if headers is None else headers
    if isinstance(data, dict): data = json.dumps(data)
    ret_headers = {'status' : 'ok'}
    ret_headers.update(passed_headers)
    return server_responce(ret_headers, data)
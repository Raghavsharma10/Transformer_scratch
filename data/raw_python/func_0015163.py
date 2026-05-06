def _process_req_txt(req):
    '''Returns a processed request or raises an exception'''
    if req.status_code == 404:
        return ''
    if req.status_code != 200:
        raise DapiCommError('Response of the server was {code}'.format(code=req.status_code))
    return req.text
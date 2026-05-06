def safe_request(fct):
    ''' Return json messages instead of raising errors '''
    def inner(*args, **kwargs):
        ''' decorator '''
        try:
            _data = fct(*args, **kwargs)
        except requests.exceptions.ConnectionError as error:
            return {'error': str(error), 'status': 404}

        if _data.ok:
            if _data.content:
                safe_data = _data.json()
            else:
                safe_data = {'success': True}
        else:
            safe_data = {'error': _data.reason, 'status': _data.status_code}

        return safe_data
    return inner
def capakey_rest_gateway_request(url, headers={}, params={}):
    '''
    Utility function that helps making requests to the CAPAKEY REST service.

    :param string url: URL to request.
    :param dict headers: Headers to send with the URL.
    :param dict params: Parameters to send with the URL.
    :returns: Result of the call.
    '''
    try:
        res = requests.get(url, headers=headers, params=params)
        res.raise_for_status()
        return res
    except requests.ConnectionError as ce:
        raise GatewayRuntimeException(
            'Could not execute request due to connection problems:\n%s' % repr(ce),
            ce
        )
    except requests.HTTPError as he:
        raise GatewayResourceNotFoundException()
    except requests.RequestException as re:
        raise GatewayRuntimeException(
            'Could not execute request due to:\n%s' % repr(re),
            re
        )
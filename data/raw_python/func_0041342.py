def _make_json_result(code, message="", results=None):
    """
    An utility method to prepare a JSON result string, usable by the
    SignalReceiver

    :param code: A HTTP Code
    :param message: An associated message
    """
    return code, json.dumps({'code': code,
                             'message': message,
                             'results': results})
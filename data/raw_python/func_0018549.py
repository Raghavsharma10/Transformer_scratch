def query(request):
    """Query encoder/decoder with a request value"""

    def inner(func, obj):
        result_code = func(obj, request)

        if result_code is not constants.OK:
            raise OpusError(result_code)

        return result_code

    return inner
def set(request):
    """Set new CTL value to a encoder/decoder"""

    def inner(func, obj, value):
        result_code = func(obj, request, value)
        if result_code is not constants.OK:
            raise OpusError(result_code)

    return inner
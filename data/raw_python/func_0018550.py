def get(request, result_type):
    """Get CTL value from a encoder/decoder"""

    def inner(func, obj):
        result = result_type()
        result_code = func(obj, request, ctypes.byref(result))

        if result_code is not constants.OK:
            raise OpusError(result_code)

        return result.value

    return inner
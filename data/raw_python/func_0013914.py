def xml_request(check_object=False, check_invalid_data_mover=False):
    """ indicate the return value is a xml api request

    :param check_invalid_data_mover:
    :param check_object:
    :return: the response of this request
    """

    def decorator(f):
        @functools.wraps(f)
        def func_wrapper(self, *argv, **kwargs):
            request = f(self, *argv, **kwargs)
            return self.request(
                request, check_object=check_object,
                check_invalid_data_mover=check_invalid_data_mover)

        return func_wrapper

    return decorator
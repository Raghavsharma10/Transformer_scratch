def parse(cls, payload):
        """ Parse client request """
        try:
            method, args, ref = payload
        except Exception as exception:
            raise RequestParseError(exception)
        else:
            return method, args, ref
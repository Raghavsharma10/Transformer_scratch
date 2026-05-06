def basic_parse(response, buf_size=ijson.backend.BUFSIZE):
        """
        Iterator yielding unprefixed events.

        Parameters:

        - response: a stream response from requests
        """
        lexer = iter(IncrementalJsonParser.lexer(response, buf_size))
        for value in ijson.backend.parse_value(lexer):
            yield value
        try:
            next(lexer)
        except StopIteration:
            pass
        else:
            raise ijson.common.JSONError('Additional data')
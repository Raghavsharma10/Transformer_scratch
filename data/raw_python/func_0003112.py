def do_inspect(self, code, cursor_pos, detail_level=0):
        """
        Method called on help requests
        """
        self._klog.info("{%s}", code[cursor_pos:cursor_pos+10])

        # Find the token for which help is requested
        token, start = token_at_cursor(code, cursor_pos)
        self._klog.debug("token={%s} {%d}", token, detail_level)

        # Find the help for this token
        if not is_magic(token, start, code):
            info = sparql_help.get(token.upper(), None)
        elif token == '%':
            info = magic_help
        else:
            info = magics.get(token, None)
            if info:
                info = '{} {}\n\n{}'.format(token, *info)

        return {'status': 'ok',
                'data': {'text/plain': info},
                'metadata': {},
                'found': info is not None
               }
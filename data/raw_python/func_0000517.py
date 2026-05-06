def split_string(self, string, splitter='.', allow_empty=True):
        """Split the string with respect of quotes"""
        i = 0
        rv = []
        need_split = False
        while i < len(string):
            m = re.compile(_KEY_NAME).match(string, i)
            if not need_split and m:
                i = m.end()
                body = m.group(1)
                if body[:3] == '"""':
                    body = self.converter.unescape(body[3:-3])
                elif body[:3] == "'''":
                    body = body[3:-3]
                elif body[0] == '"':
                    body = self.converter.unescape(body[1:-1])
                elif body[0] == "'":
                    body = body[1:-1]
                if not allow_empty and not body:
                    raise TomlDecodeError(
                        self.lineno,
                        'Empty section name is not allowed: %r' % string)
                rv.append(body)
                need_split = True
            elif need_split and string[i] == splitter:
                need_split = False
                i += 1
                continue
            else:
                raise TomlDecodeError(self.lineno,
                                      'Illegal section name: %r' % string)
        if not need_split:
            raise TomlDecodeError(
                self.lineno,
                'Empty section name is not allowed: %r' % string)
        return rv
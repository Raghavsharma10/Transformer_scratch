def get_cursor_position(self):
        """Returns the terminal (row, column) of the cursor

        0-indexed, like blessings cursor positions"""
        # TODO would this be cleaner as a parameter?
        in_stream = self.in_stream

        query_cursor_position = u"\x1b[6n"
        self.write(query_cursor_position)

        def retrying_read():
            while True:
                try:
                    c = in_stream.read(1)
                    if c == '':
                        raise ValueError("Stream should be blocking - should't"
                                         " return ''. Returned %r so far",
                                         (resp,))
                    return c
                except IOError:
                    raise ValueError(
                        'cursor get pos response read interrupted'
                    )
                    # find out if this ever really happens - if so, continue

        resp = ''
        while True:
            c = retrying_read()
            resp += c
            m = re.search('(?P<extra>.*)'
                          '(?P<CSI>\x1b\[|\x9b)'
                          '(?P<row>\\d+);(?P<column>\\d+)R', resp, re.DOTALL)
            if m:
                row = int(m.groupdict()['row'])
                col = int(m.groupdict()['column'])
                extra = m.groupdict()['extra']
                if extra:
                    if self.extra_bytes_callback:
                        self.extra_bytes_callback(
                            extra.encode(in_stream.encoding)
                        )
                    else:
                        raise ValueError(("Bytes preceding cursor position "
                                          "query response thrown out:\n%r\n"
                                          "Pass an extra_bytes_callback to "
                                          "CursorAwareWindow to prevent this")
                                         % (extra,))
                return (row - 1, col - 1)
def diff(cls, a, b, ignore_formatting=False):
        """Returns two FSArrays with differences underlined"""
        def underline(x): return u'\x1b[4m%s\x1b[0m' % (x,)
        def blink(x): return u'\x1b[5m%s\x1b[0m' % (x,)
        a_rows = []
        b_rows = []
        max_width = max([len(row) for row in a] + [len(row) for row in b])
        a_lengths = []
        b_lengths = []
        for a_row, b_row in zip(a, b):
            a_lengths.append(len(a_row))
            b_lengths.append(len(b_row))
            extra_a = u'`' * (max_width - len(a_row))
            extra_b = u'`' * (max_width - len(b_row))
            a_line = u''
            b_line = u''
            for a_char, b_char in zip(a_row + extra_a, b_row + extra_b):
                if ignore_formatting:
                    a_char_for_eval = a_char.s if isinstance(a_char, FmtStr) else a_char
                    b_char_for_eval = b_char.s if isinstance(b_char, FmtStr) else b_char
                else:
                    a_char_for_eval = a_char
                    b_char_for_eval = b_char
                if a_char_for_eval == b_char_for_eval:
                    a_line += actualize(a_char)
                    b_line += actualize(b_char)
                else:
                    a_line += underline(blink(actualize(a_char)))
                    b_line += underline(blink(actualize(b_char)))
            a_rows.append(a_line)
            b_rows.append(b_line)
        hdiff = '\n'.join(a_line + u' %3d | %3d ' % (a_len, b_len) + b_line for a_line, b_line, a_len, b_len in zip(a_rows, b_rows, a_lengths, b_lengths))
        return hdiff
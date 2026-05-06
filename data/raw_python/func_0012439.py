def _backtick_columns(cols):
        """
        Quote the column names
        """
        def bt(s):
            b = '' if s == '*' or not s else '`'
            return [_ for _ in [b + (s or '') + b] if _]

        formatted = []
        for c in cols:
            if c[0] == '#':
                formatted.append(c[1:])
            elif c.startswith('(') and c.endswith(')'):
                # WHERE (column_a, column_b) IN ((1,10), (1,20))
                formatted.append(c)
            else:
                # backtick the former part when it meets the first dot, and then all the rest
                formatted.append('.'.join(bt(c.split('.')[0]) + bt('.'.join(c.split('.')[1:]))))

        return ', '.join(formatted)
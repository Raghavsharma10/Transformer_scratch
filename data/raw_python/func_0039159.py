def parse_markdown(self, attr_string):
        """Read markdown attributes."""
        attr_string = attr_string.strip('{}')
        splitter = re.compile(self.split_regex(separator=self.spnl))
        attrs = splitter.split(attr_string)[1::2]

        # match single word attributes e.g. ```python
        if len(attrs) == 1 \
                and not attr_string.startswith(('#', '.')) \
                and '=' not in attr_string:
            return '', [attr_string], OrderedDict()

        try:
            id = [a[1:] for a in attrs if a.startswith('#')][0]
        except IndexError:
            id = ''

        classes = [a[1:] for a in attrs if a.startswith('.')]
        special = ['unnumbered' for a in attrs if a == '-']
        classes.extend(special)

        kvs = OrderedDict(a.split('=', 1) for a in attrs if '=' in a)

        return id, classes, kvs
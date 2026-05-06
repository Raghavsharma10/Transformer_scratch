def generate(self, nameformat=None, capitalize=None, formatters=None, **kwargs):
        '''Pick a random name form a specified list of name parts'''

        nameformat = nameformat or self.nameformat
        capitalize = capitalize or self.capitalize
        formatters = formatters or {}

        lines = self._get_lines(kwargs)
        names = dict((k, v['name']) for k, v in list(lines.items()))

        if capitalize:
            names = dict((k, n.capitalize()) for k, n in list(names.items()))

        merged_formatters = dict()

        try:
            merged_formatters = dict(
                (k, self.formatters.get(k, []) + formatters.get(k, [])) for k in set(list(self.formatters.keys()) + list(formatters.keys()))
            )
        except AttributeError:
            raise TypeError("keyword argument 'formatters' for Censusname.generate() must be a dict")

        if merged_formatters:
            for key, functions in list(merged_formatters.items()):
                # 'surname', [func_a, func_b]
                for func in functions:
                    # names['surname'] = func_a(name['surname'])
                    names[key] = func(names[key])

        return nameformat.format(**names)
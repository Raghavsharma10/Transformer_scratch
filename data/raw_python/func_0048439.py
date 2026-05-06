def __generate(self):
        """Generates dates patterns"""
        base = []
        texted = []
        for pat in ALL_PATTERNS:
            data = pat.copy()
            data['pattern'] = data['pattern']
            data['right'] = True
            data['basekey'] = data['key']
            base.append(data)

            data = pat.copy()
            data['basekey'] = data['key']
            data['key'] += ':time_1'
            data['right'] = True
            data['pattern'] = data['pattern'] + Optional(Literal(",")).suppress() + BASE_TIME_PATTERNS['pat:time:minutes']
            data['time_format'] = '%H:%M'
            data['length'] = {'min' : data['length']['min'] + 5, 'max' : data['length']['max'] + 8}
            base.append(data)

            data = pat.copy()
            data['basekey'] = data['key']
            data['right'] = True
            data['key'] += ':time_2'
            data['pattern'] = data['pattern'] + Optional(oneOf([',', '|'])).suppress() + BASE_TIME_PATTERNS['pat:time:full']
            data['time_format'] = '%H:%M:%S'
            data['length'] = {'min' : data['length']['min'] + 9, 'max' : data['length']['max'] + 9}
            base.append(data)

        for pat in base:
            # Right
            data = pat.copy()
            data['key'] += ':t_right'
            data['pattern'] = lineStart + data['pattern'] +  Optional(oneOf([',', '|', ':', ')'])).suppress() + restOfLine.suppress()
            data['length'] = {'min' : data['length']['min'] + 1, 'max' : data['length']['max'] + 90}
            texted.append(data)

        base.extend(texted)
        self.patterns = base
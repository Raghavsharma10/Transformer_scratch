def parse_object(self, data):
        """ Look for datetime looking strings. """
        for key, value in data.items():
            if isinstance(value, (str, type(u''))) and \
               self.strict_iso_match.match(value):
                data[key] = dateutil.parser.parse(value)
        return data
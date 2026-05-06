def get_parsed_value(self, value):
        """
        Helper to cast string to datetime using :member:`parse_format`.

        :param value: String representing a datetime
        :type value: str
        :return: datetime
        """

        def get_parser(parser_desc):
            try:
                return parser_desc['parser']
            except TypeError:
                try:
                    return get_parser(self.date_parsers[parser_desc])
                except KeyError:
                    return parser_desc
            except KeyError:
                pass

        parser = get_parser(self.parse_format)

        if parser is None:
            try:
                return dateutil_parse(value)
            except ValueError:
                return None

        if callable(parser):
            return parser(value)
        return datetime.strptime(value, parser)
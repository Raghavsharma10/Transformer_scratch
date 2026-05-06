def get_formatted_value(self, value):
        """
        Returns a string from datetime using :member:`parse_format`.

        :param value: Datetime to cast to string
        :type value: datetime
        :return: str
        """

        def get_formatter(parser_desc):
            try:
                return parser_desc['formatter']
            except TypeError:
                if isinstance(parser_desc, str):
                    try:
                        return get_formatter(self.date_parsers[parser_desc])
                    except KeyError:
                        return parser_desc
                else:
                    pass
            except KeyError:
                try:
                    if isinstance(parser_desc['parser'], str):
                        return parser_desc['parser']
                except KeyError:
                    pass

        formatter = get_formatter(self.parse_format)

        if formatter is None:
            return str(value)

        if callable(formatter):
            return formatter(value)

        return value.strftime(format=formatter)
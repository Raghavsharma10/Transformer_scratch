def parse_field(cls: Type[DocumentType], field_name: str, line: str) -> Any:
        """
        Parse a document field with regular expression and return the value

        :param field_name: Name of the field
        :param line: Line string to parse
        :return:
        """
        try:
            match = cls.fields_parsers[field_name].match(line)
            if match is None:
                raise AttributeError
            value = match.group(1)
        except AttributeError:
            raise MalformedDocumentError(field_name)
        return value
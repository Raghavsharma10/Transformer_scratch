def parse_string(self):
        """
        Accesses the S-Expression parse string stored on the XML document

        :getter: Returns the parse string
        :type: str

        """
        if self._parse_string is None:
            parse_text = self._element.xpath('parse/text()')
            if len(parse_text) > 0:
                self._parse_string = parse_text[0]
        return self._parse_string
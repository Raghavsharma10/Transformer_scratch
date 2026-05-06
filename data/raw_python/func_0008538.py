def from_xml(cls, xml):
        """ Returns a new Text from the given XML string.
        """
        s = parse_string(xml)
        return Sentence(s.split("\n")[0], token=s.tags, language=s.language)
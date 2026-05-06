def xml(self):
        """ Yields the sentence as an XML-formatted string (plain bytestring, UTF-8 encoded).
            All the sentences in the XML are wrapped in a <text> element.
        """
        xml = []
        xml.append('<?xml version="1.0" encoding="%s"?>' % XML_ENCODING.get(self.encoding, self.encoding))
        xml.append("<%s>" % XML_TEXT)
        xml.extend([sentence.xml for sentence in self])
        xml.append("</%s>" % XML_TEXT)
        return "\n".join(xml)
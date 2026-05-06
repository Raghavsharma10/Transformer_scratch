def xml(self) -> _BaseXML:
        """Unicode representation of the XML content
        (`learn more <http://www.diveintopython3.net/strings.html>`_).
        """
        if self._xml:
            return self.raw_xml.decode(self.encoding)
        else:
            return etree.tostring(self.element, encoding='unicode').strip()
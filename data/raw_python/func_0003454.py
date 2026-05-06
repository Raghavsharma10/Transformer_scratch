def raw_xml(self) -> _RawXML:
        """Bytes representation of the XML content.
        (`learn more <http://www.diveintopython3.net/strings.html>`_).
        """
        if self._xml:
            return self._xml
        else:
            return etree.tostring(self.element, encoding='unicode').strip().encode(self.encoding)
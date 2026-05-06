def lxml(self) -> _LXML:
        """`lxml <http://lxml.de>`_ representation of the
        :class:`Element <Element>` or :class:`XML <XML>`.
        """
        if self._lxml is None:
            self._lxml = etree.fromstring(self.raw_xml)

        return self._lxml
def encoding(self) -> _Encoding:
        """The encoding string to be used, extracted from the XML and
        :class:`XMLResponse <XMLResponse>` header.
        """
        if self._encoding:
            return self._encoding

        # Scan meta tags for charset.
        if self._xml:
            self._encoding = html_to_unicode(self.default_encoding, self._xml)[0]

        return self._encoding if self._encoding else self.default_encoding
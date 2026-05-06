def json(self, conversion: _Text = 'badgerfish') -> Mapping:
        """A JSON Representation of the XML.  Default is badgerfish.
        :param conversion: Which conversion method to use. (`learn more <https://github.com/sanand0/xmljson#conventions>`_)
        """
        if not self._json:

            if conversion is 'badgerfish':
                from xmljson import badgerfish as serializer

            elif conversion is 'abdera':
                from xmljson import abdera as serializer

            elif conversion is 'cobra':
                from xmljson import cobra as serializer

            elif conversion is 'gdata':
                from xmljson import gdata as serializer

            elif conversion is 'parker':
                from xmljson import parker as serializer

            elif conversion is 'yahoo':
                from xmljson import yahoo as serializer

            self._json = json.dumps(serializer.data(etree.fromstring(self.xml)))

        return self._json
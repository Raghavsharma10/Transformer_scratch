def _parse_text(self, element_name, namespace=''):
        """
        Returns the text, as a string, of the specified element in the specified
        namespace of the RSS feed.

        Takes element_name and namespace as strings.
        """
        try:
            text = self._channel.find('.//' + namespace + element_name).text
        except AttributeError:
            raise Exception(
                'Element, {0} not found in RSS feed'.format(element_name)
            )

        return text
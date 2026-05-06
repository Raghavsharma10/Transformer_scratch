def fill(self, field, value):
        """
        Fill a specified form field in the current document.

        :param field: an instance of :class:`zombie.dom.DOMNode`
        :param value: any string value
        :return: self to allow function chaining.
        """
        self.client.nowait('browser.fill', (field, value))
        return self
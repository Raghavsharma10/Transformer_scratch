def value(self, value):
        """
        Used to set the ``value`` of form elements.
        """
        self.client.nowait(
            'set_field', (Literal('browser'), self.element, value))
def as_dict(self):
        """
        Serializes the object necessary data in a dictionary.

        :returns: Serialized data in a dictionary.
        :rtype: dict
        """

        element_dict = super(HTMLElement, self).as_dict()

        if hasattr(self, 'content'):
            element_dict['content'] = self.content

        return element_dict
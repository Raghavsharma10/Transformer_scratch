def as_dict(self):
        """
        Serializes the object necessary data in a dictionary.

        :returns: Serialized data in a dictionary.
        :rtype: dict
        """

        element_dict = dict()
        if hasattr(self, 'namespace'):
            element_dict['namespace'] = self.namespace
        if hasattr(self, 'name'):
            element_dict['name'] = self.name
        if hasattr(self, 'text'):
            element_dict['text'] = self.text

        attr_dict = dict()
        for attr in self.attrs:
            if hasattr(self, attr):
                attr_dict[attr] = getattr(self, attr)
        element_dict['attrs'] = attr_dict

        return element_dict
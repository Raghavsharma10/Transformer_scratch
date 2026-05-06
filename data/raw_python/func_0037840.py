def _items(self, type_filter=None, name=None):
        """
        Args:
            type_filter(list): Optional iterable of types to return (GroupDict only)
            name(str): Only return key by this name

        Alternative generator for items() method
        """

        if name:
            if type_filter and self._key_attr == 'type':
                if name in type_filter and name in self:
                    yield name, self[name]
            elif name in self:
                yield name, self[name]

        elif type_filter and self._key_attr == 'type':
            for key, val in self.items():
                if key in type_filter:
                    yield key, val
        else:
            for key, val in self.items():
                yield key, val
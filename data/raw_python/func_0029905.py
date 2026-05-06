def partital_dict(self, with_name=True):
        """Returns the name as a dict, but with only the items that are
        particular to a PartitionName."""

        d = self._dict(with_name=False)

        d = {k: d.get(k) for k, _, _ in PartialPartitionName._name_parts if d.get(k, False)}

        if 'format' in d and d['format'] == Name.DEFAULT_FORMAT:
            del d['format']

        d['name'] = self.name

        return d
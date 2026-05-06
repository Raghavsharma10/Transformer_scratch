def parse_pandoc(self, attrs):
        """Read pandoc attributes."""
        id = attrs[0]
        classes = attrs[1]
        kvs = OrderedDict(attrs[2])

        return id, classes, kvs
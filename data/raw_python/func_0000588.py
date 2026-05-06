def populate(self, **values):
        """Populate values to fields. Skip non-existing."""
        values = values.copy()
        fields = list(self.iterate_with_name())
        for _, structure_name, field in fields:
            if structure_name in values:
                field.__set__(self, values.pop(structure_name))
        for name, _, field in fields:
            if name in values:
                field.__set__(self, values.pop(name))
def parse(self, data):
        """Parse operator and value from filter's data."""
        val = data.get(self.name, missing)
        if not isinstance(val, dict):
            return (self.operators['$eq'], self.field.deserialize(val)),

        return tuple(
            (
                self.operators[op],
                (self.field.deserialize(val)) if op not in self.list_ops else [
                    self.field.deserialize(v) for v in val])
            for (op, val) in val.items() if op in self.operators
        )
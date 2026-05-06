def select_by_field(self, base, field, value):
        """Return collection of acces whose field equal value"""
        Ac = self.ACCES
        return groups.Collection(Ac(base, i) for i, row in self.items() if row[field] == value)
def columns(self):
        """Return names of all the addressable columns (including foreign keys) referenced in user supplied model"""
        res = [col['name'] for col in self.column_definitions]
        res.extend([col['name'] for col in self.foreign_key_definitions])
        return res
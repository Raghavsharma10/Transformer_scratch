def add_relations(self, relations):
        """Add multiple relations to a bijection"""
        for source, destination in relations:
            self.add_relation(source, destination)
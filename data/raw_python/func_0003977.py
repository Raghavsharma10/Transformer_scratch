def copy_with_new_relations(self, new_relations):
        """Create a new match object extended with new relations"""
        result = self.__class__(self.forward.items())
        result.add_relations(new_relations.items())
        result.previous_ends1 = set(new_relations.values())
        return result
def get_by_slot(self, parent_object, slot):
        """
        Return a placeholder by key.
        """
        placeholder = self.parent(parent_object).get(slot=slot)
        placeholder.parent = parent_object  # fill the reverse cache
        return placeholder
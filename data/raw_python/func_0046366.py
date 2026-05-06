def set_children(self, child_ids):
        """Set the children IDs"""
        if not self._supports_simple_sequencing():
            raise errors.IllegalState()
        self._my_map['childIds'] = [str(i) for i in child_ids]
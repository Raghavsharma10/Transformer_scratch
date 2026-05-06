def _adopt_orphans(self, negligent_parent_id):
        """Clean up orphaned children"""
        for child_id in self._hts.get_children(negligent_parent_id):
            self.remove_child(negligent_parent_id, child_id)
            if not self._hts.has_parents(child_id):
                self._assign_as_root(child_id)
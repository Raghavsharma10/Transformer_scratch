def get_child_ids(self):
        """gets the ids for the child parts"""
        if self.has_magic_children():
            if self._child_parts is None:
                self.generate_children()
            child_ids = list()
            for part in self._child_parts:
                child_ids.append(part.get_id())
            return IdList(child_ids,
                          runtime=self.my_osid_object._runtime,
                          proxy=self.my_osid_object._runtime)
        raise IllegalState()
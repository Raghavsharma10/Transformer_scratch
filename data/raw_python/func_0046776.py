def get_children(self):
        """return the current child parts of this assessment part"""
        if self.has_magic_children():
            if self._child_parts is None:
                self.generate_children()
            return self._child_parts
        raise IllegalState()
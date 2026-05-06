def copy(self):
        """Return a new instance, deep-copying all the attributes."""
        o = self.__class__(self.project, self.name)
        Scriptable.copy(self, o)
        o.position = tuple(self.position)
        o.direction = self.direction
        o.rotation_style = self.rotation_style
        o.size = self.size
        o.is_draggable = self.is_draggable
        o.is_visible = self.is_visible
        return o
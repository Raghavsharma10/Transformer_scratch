def pos(self):
        """ The position of this event in the local coordinate system of the
        visual.
        """
        if self._pos is None:
            tr = self.visual.get_transform('canvas', 'visual')
            self._pos = tr.map(self.mouse_event.pos)
        return self._pos
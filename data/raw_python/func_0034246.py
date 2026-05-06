def reset(self):
        """
        Clears `nick` and `own_ids`, sets `center` to `world.center`,
        and then calls `cells_changed()`.
        """
        self.own_ids.clear()
        self.nick = ''
        self.center = self.world.center
        self.cells_changed()
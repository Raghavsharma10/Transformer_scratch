def trail(self):
        """ Return an (N, 2) array of mouse coordinates for every event in the
        current mouse drag operation.

        Returns None if there is no current drag operation.
        """
        events = self.drag_events()
        if events is None:
            return None

        trail = np.empty((len(events), 2), dtype=int)
        for i, ev in enumerate(events):
            trail[i] = ev.pos

        return trail
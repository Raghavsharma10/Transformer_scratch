def label(self, t):
        """Get the label of the song at a given time in seconds
        """
        if self.labels is None:
            return None
        prev_label = None
        for l in self.labels:
            if l.time > t:
                break
            prev_label = l
        if prev_label is None:
            return None
        return prev_label.name
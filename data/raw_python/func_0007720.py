def clear(self):
        """
        Clear the segment.
        :return: None
        """
        for _, frame in self._segments.items():
            frame.configure(background=self._bg_color)
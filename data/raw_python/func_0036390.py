def current_frame(self, n):
        """Sets current frame to ``n``

        :param integer n: Frame to set to ``current_frame``
        """
        self.sound.seek(n)
        self._current_frame = n
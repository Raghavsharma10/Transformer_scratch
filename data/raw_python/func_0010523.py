def on_update(self, value, *args, **kwargs):
        """
        Inform the parent of progress.
        :param value: The value of this subprogresscallback
        :param args: Extra positional arguments
        :param kwargs: Extra keyword arguments
        """
        parent_value = self._parent_min
        if self._max != self._min:
            sub_progress = (value - self._min) / (self._max - self._min)
            parent_value = self._parent_min + sub_progress * (self._parent_max - self._parent_min)
        self._parent.update(parent_value, *args, **kwargs)
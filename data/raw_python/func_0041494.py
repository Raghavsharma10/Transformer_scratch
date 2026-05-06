def is_equivalent(self, callback, details_filter=None):
        """Check if the callback provided is the same as the internal one.

        :param callback: callback used for comparison
        :param details_filter: callback used for comparison
        :returns: false if not the same callback, otherwise true
        :rtype: boolean
        """
        cb = self.callback
        if cb is None and callback is not None:
            return False
        if cb is not None and callback is None:
            return False
        if cb is not None and callback is not None \
           and not reflection.is_same_callback(cb, callback):
            return False
        if details_filter is not None:
            if self._details_filter is None:
                return False
            else:
                return reflection.is_same_callback(self._details_filter,
                                                   details_filter)
        else:
            return self._details_filter is None
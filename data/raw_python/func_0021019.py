def _normalize_slice(self, index, pipe=None):
        """Given a :obj:`slice` *index*, return a 4-tuple
        ``(start, stop, step, fowrward)``. The first three items can be used
        with the ``range`` function to retrieve the values associated with the
        slice; the last item indicates the direction.
        """
        if index.step == 0:
            raise ValueError
        pipe = self.redis if pipe is None else pipe

        len_self = self.__len__(pipe)

        step = index.step or 1
        forward = step > 0
        step = abs(step)

        if index.start is None:
            start = 0 if forward else len_self - 1
        elif index.start < 0:
            start = max(len_self + index.start, 0)
        else:
            start = min(index.start, len_self)

        if index.stop is None:
            stop = len_self if forward else -1
        elif index.stop < 0:
            stop = max(len_self + index.stop, 0)
        else:
            stop = min(index.stop, len_self)

        if not forward:
            start, stop = min(stop + 1, len_self), min(start + 1, len_self)

        return start, stop, step, forward, len_self
def get(self, block=True, timeout=None):
        """Remove and return an item from the queue.

        If optional args ``block`` is ``True`` and ``timeout`` is
        ``None`` (the default), block if necessary until an item is
        available.  If ``timeout`` is a positive number, it blocks at
        most ``timeout`` seconds and raises the :exc:`Queue.Empty` exception
        if no item was available within that time. Otherwise (``block`` is
        ``False``), return an item if one is immediately available, else
        raise the :exc:`Queue.Empty` exception (``timeout`` is ignored
        in that case).

        """
        if not block:
            return self.get_nowait()
        item = self._bpop([self.name], timeout=timeout)
        if item is not None:
            return item
        raise Empty
def full(self):
        """Return ``True`` if the queue is full, ``False``
        otherwise (not reliable!).

        Only applicable if :attr:`maxsize` is set.

        """
        return self.maxsize and len(self.list) >= self.maxsize or False
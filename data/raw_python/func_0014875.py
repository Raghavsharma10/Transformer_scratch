def get1(self, name, **kwargs):
        """
        Look up gender for a single name.
        See :py:meth:`get`.
        Doesn't support retheader option.
        """
        if 'retheader' in kwargs:
            raise GenderizeException(
                "get1() doesn't support the retheader option.")
        return self.get([name], **kwargs)[0]